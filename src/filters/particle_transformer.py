"""Particle Transformer resampling and DPF variant."""

from __future__ import annotations

import tensorflow as tf

from src.filters.dpf import DPFBase


class ParticleTransformerResampler(tf.keras.layers.Layer):
    """Particle Transformer resampler aligned with Zhu et al. (2020)."""

    class _WeightedMultiHeadAttention(tf.keras.layers.Layer):
        """Multi-head attention with additive log-weight bias on key positions."""

        def __init__(
            self,
            d_model: int,
            num_heads: int,
            dropout_rate: float = 0.0,
        ) -> None:
            super().__init__()
            if int(d_model) % int(num_heads) != 0:
                raise ValueError("d_model must be divisible by num_heads.")

            self.d_model = int(d_model)
            self.num_heads = int(num_heads)
            self.head_dim = self.d_model // self.num_heads

            self.w_q = tf.keras.layers.Dense(self.d_model, use_bias=False)
            self.w_k = tf.keras.layers.Dense(self.d_model, use_bias=False)
            self.w_v = tf.keras.layers.Dense(self.d_model, use_bias=False)
            self.w_o = tf.keras.layers.Dense(self.d_model, use_bias=False)
            self.dropout = tf.keras.layers.Dropout(float(dropout_rate))

        def _split_heads(self, x: tf.Tensor) -> tf.Tensor:
            b = tf.shape(x)[0]
            t = tf.shape(x)[1]
            x = tf.reshape(x, [b, t, self.num_heads, self.head_dim])
            return tf.transpose(x, perm=[0, 2, 1, 3])

        def _merge_heads(self, x: tf.Tensor) -> tf.Tensor:
            b = tf.shape(x)[0]
            t = tf.shape(x)[2]
            x = tf.transpose(x, perm=[0, 2, 1, 3])
            return tf.reshape(x, [b, t, self.d_model])

        def call(
            self,
            query: tf.Tensor,
            key_value: tf.Tensor,
            log_w_keys: tf.Tensor | None,
            training: bool | None = None,
        ) -> tuple[tf.Tensor, tf.Tensor]:
            q = self._split_heads(self.w_q(query))
            k = self._split_heads(self.w_k(key_value))
            v = self._split_heads(self.w_v(key_value))

            scale = tf.math.rsqrt(tf.cast(self.head_dim, q.dtype))
            logits = tf.einsum("bhqd,bhkd->bhqk", q, k) * scale

            if log_w_keys is not None:
                log_w_keys = tf.convert_to_tensor(log_w_keys, dtype=logits.dtype)
                logits = logits + log_w_keys[:, tf.newaxis, tf.newaxis, :]

            attn = tf.nn.softmax(logits, axis=-1)
            attn = self.dropout(attn, training=training)

            out = tf.einsum("bhqk,bhkd->bhqd", attn, v)
            out = self._merge_heads(out)
            out = self.w_o(out)
            return out, attn

    class _EncoderBlock(tf.keras.layers.Layer):
        """Weighted self-attention encoder block."""

        def __init__(
            self,
            d_model: int,
            num_heads: int,
            ff_hidden: int,
            dropout_rate: float = 0.0,
        ) -> None:
            super().__init__()
            self.self_attn = ParticleTransformerResampler._WeightedMultiHeadAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
            )
            self.ffn = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(int(ff_hidden), activation="relu"),
                    tf.keras.layers.Dropout(float(dropout_rate)),
                    tf.keras.layers.Dense(int(d_model)),
                ]
            )
            self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.dropout1 = tf.keras.layers.Dropout(float(dropout_rate))
            self.dropout2 = tf.keras.layers.Dropout(float(dropout_rate))

        def call(
            self,
            x: tf.Tensor,
            log_w: tf.Tensor,
            training: bool | None = None,
        ) -> tf.Tensor:
            attn_out, _ = self.self_attn(
                query=x,
                key_value=x,
                log_w_keys=log_w,
                training=training,
            )
            x = self.norm1(x + self.dropout1(attn_out, training=training))
            ffn_out = self.ffn(x, training=training)
            x = self.norm2(x + self.dropout2(ffn_out, training=training))
            return x

    class _DecoderBlock(tf.keras.layers.Layer):
        """Decoder block with seed self-attn then weighted cross-attn."""

        def __init__(
            self,
            d_model: int,
            num_heads: int,
            ff_hidden: int,
            dropout_rate: float = 0.0,
        ) -> None:
            super().__init__()
            self.self_attn = ParticleTransformerResampler._WeightedMultiHeadAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
            )
            self.cross_attn = ParticleTransformerResampler._WeightedMultiHeadAttention(
                d_model=d_model,
                num_heads=num_heads,
                dropout_rate=dropout_rate,
            )
            self.ffn = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(int(ff_hidden), activation="relu"),
                    tf.keras.layers.Dropout(float(dropout_rate)),
                    tf.keras.layers.Dense(int(d_model)),
                ]
            )
            self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.dropout1 = tf.keras.layers.Dropout(float(dropout_rate))
            self.dropout2 = tf.keras.layers.Dropout(float(dropout_rate))
            self.dropout3 = tf.keras.layers.Dropout(float(dropout_rate))

        def call(
            self,
            query: tf.Tensor,
            memory: tf.Tensor,
            log_w_memory: tf.Tensor,
            training: bool | None = None,
        ) -> tuple[tf.Tensor, tf.Tensor]:
            self_out, _ = self.self_attn(
                query=query,
                key_value=query,
                log_w_keys=None,
                training=training,
            )
            query = self.norm1(query + self.dropout1(self_out, training=training))

            cross_out, cross_attn = self.cross_attn(
                query=query,
                key_value=memory,
                log_w_keys=log_w_memory,
                training=training,
            )
            query = self.norm2(query + self.dropout2(cross_out, training=training))

            ffn_out = self.ffn(query, training=training)
            query = self.norm3(query + self.dropout3(ffn_out, training=training))
            return query, cross_attn

    def __init__(
        self,
        num_particles: int,
        d_model: int = 128,
        hidden: int = 128,
        num_heads: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 1,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_particles = int(num_particles)
        self.d_model = int(d_model)
        self.hidden = int(hidden)
        self.num_heads = int(num_heads)
        self.num_encoder_layers = int(num_encoder_layers)
        self.num_decoder_layers = int(num_decoder_layers)
        self.dropout_rate = float(dropout_rate)

        self.input_proj = tf.keras.layers.Dense(self.d_model)
        self.encoder_blocks = [
            self._EncoderBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                ff_hidden=self.hidden,
                dropout_rate=self.dropout_rate,
            )
            for _ in range(self.num_encoder_layers)
        ]
        self.decoder_blocks = [
            self._DecoderBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                ff_hidden=self.hidden,
                dropout_rate=self.dropout_rate,
            )
            for _ in range(self.num_decoder_layers)
        ]

        self.final_proj: tf.keras.layers.Layer | None = None
        self.seed_queries: tf.Variable | None = None

    def build(self, input_shape) -> None:
        if len(input_shape) != 3:
            raise ValueError("x must have shape [B, N, dx].")
        dx = int(input_shape[-1])
        self.seed_queries = self.add_weight(
            name="seed_queries",
            shape=(self.num_particles, self.d_model),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.final_proj = tf.keras.layers.Dense(dx)
        super().build(input_shape)

    def _scale_to_unit_box(self, x: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        x_min = tf.reduce_min(x, axis=1, keepdims=True)
        x_max = tf.reduce_max(x, axis=1, keepdims=True)
        x_range = tf.maximum(x_max - x_min, tf.constant(1e-6, dtype=x.dtype))
        x_scaled = ((x - x_min) / x_range) * 2.0 - 1.0
        return x_scaled, x_min, x_range

    def _undo_scale(self, x_scaled: tf.Tensor, x_min: tf.Tensor, x_range: tf.Tensor) -> tf.Tensor:
        return (x_scaled + 1.0) * 0.5 * x_range + x_min

    def call(self, x: tf.Tensor, log_w: tf.Tensor, training: bool | None = None):
        """Generate resampled particles and return final cross-attention map.

        Shapes:
          x: [B, N, dx]
          log_w: [B, N]
        Returns:
          x_new: [B, N, dx]
          attn_last: [B, N, N] (head-averaged final decoder cross-attention)
        """
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        log_w = tf.convert_to_tensor(log_w, dtype=x.dtype)
        log_w = log_w - tf.reduce_logsumexp(log_w, axis=-1, keepdims=True)

        x_scaled, x_min, x_range = self._scale_to_unit_box(x)
        tokens = self.input_proj(x_scaled)

        for block in self.encoder_blocks:
            tokens = block(tokens, log_w=log_w, training=training)

        if self.seed_queries is None:
            raise RuntimeError("seed_queries are not initialized.")
        if self.final_proj is None:
            raise RuntimeError("final_proj is not initialized.")

        batch_size = tf.shape(x)[0]
        query = tf.broadcast_to(
            self.seed_queries[tf.newaxis, :, :],
            [batch_size, self.num_particles, self.d_model],
        )

        attn_last = None
        for block in self.decoder_blocks:
            query, attn_last = block(
                query,
                memory=tokens,
                log_w_memory=log_w,
                training=training,
            )

        if attn_last is None:
            raise RuntimeError("decoder must contain at least one block.")

        attn_last = tf.reduce_mean(attn_last, axis=1)
        out_scaled = self.final_proj(query)
        x_new = self._undo_scale(out_scaled, x_min=x_min, x_range=x_range)
        return x_new, attn_last


class ParticleTransformerDPF(DPFBase):
    """Differentiable PF using a neural Particle Transformer resampler."""

    def __init__(
        self,
        ssm,
        num_particles: int = 100,
        ess_threshold: float = 0.5,
        d_model: int = 128,
        hidden: int = 128,
        num_heads: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 1,
        dropout_rate: float = 0.0,
        stop_grad_through_time: bool = True,
        resample: str | int | bool = "auto",
        debug: bool = False,
        print: bool = False,
        proposal=None,
    ) -> None:
        """Initialize ParticleTransformer DPF and its resampler network."""
        super().__init__(
            ssm,
            num_particles=num_particles,
            ess_threshold=ess_threshold,
            resample=resample,
            debug=debug,
            print=print,
            proposal=proposal,
        )
        self.d_model = int(d_model)
        self.hidden = int(hidden)
        self.num_heads = int(num_heads)
        self.num_encoder_layers = int(num_encoder_layers)
        self.num_decoder_layers = int(num_decoder_layers)
        self.dropout_rate = float(dropout_rate)
        self.stop_grad_through_time = bool(stop_grad_through_time)
        self.resampler_net = self._build_resampler()

    def _build_resampler(self) -> ParticleTransformerResampler:
        """Construct a transformer resampler with current hyperparameters."""
        self.resampler_net = ParticleTransformerResampler(
            num_particles=self.num_particles,
            d_model=self.d_model,
            hidden=self.hidden,
            num_heads=self.num_heads,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dropout_rate=self.dropout_rate,
        )
        return self.resampler_net

    def update_params(
        self,
        num_particles=None,
        ess_threshold=None,
        resample=None,
        proposal=None,
        d_model=None,
        hidden=None,
        num_heads=None,
        num_encoder_layers=None,
        num_decoder_layers=None,
        dropout_rate=None,
        stop_grad_through_time=None,
    ):
        """Update runtime hyperparameters and rebuild network if needed."""
        if num_particles is not None and int(num_particles) != self.num_particles:
            raise ValueError(
                "ParticleTransformerDPF does not support changing num_particles after "
                "initialization. Create a new instance for a different N."
            )

        super().update_params(
            num_particles=num_particles,
            ess_threshold=ess_threshold,
            resample=resample,
            proposal=proposal,
        )

        rebuild = False
        if d_model is not None and int(d_model) != self.d_model:
            self.d_model = int(d_model)
            rebuild = True
        if hidden is not None and int(hidden) != self.hidden:
            self.hidden = int(hidden)
            rebuild = True
        if num_heads is not None and int(num_heads) != self.num_heads:
            self.num_heads = int(num_heads)
            rebuild = True
        if num_encoder_layers is not None and int(num_encoder_layers) != self.num_encoder_layers:
            self.num_encoder_layers = int(num_encoder_layers)
            rebuild = True
        if num_decoder_layers is not None and int(num_decoder_layers) != self.num_decoder_layers:
            self.num_decoder_layers = int(num_decoder_layers)
            rebuild = True
        if dropout_rate is not None and float(dropout_rate) != self.dropout_rate:
            self.dropout_rate = float(dropout_rate)
            rebuild = True
        if stop_grad_through_time is not None:
            self.stop_grad_through_time = bool(stop_grad_through_time)

        if rebuild:
            self._build_resampler()

    def resample_step(self, x: tf.Tensor, log_w: tf.Tensor):
        """Resample particles with neural generator and reset to uniform weights."""
        x_new, attn = self.resampler_net(x, log_w)
        log_uniform = -tf.math.log(tf.cast(self.num_particles, x_new.dtype))
        log_w_new = tf.fill(tf.shape(log_w), log_uniform)
        parent_indices = tf.argmax(attn, axis=-1, output_type=tf.int32)
        return x_new, log_w_new, parent_indices

    def filter(
        self,
        y,
        num_particles=None,
        ess_threshold=None,
        resample=None,
        proposal=None,
        d_model=None,
        hidden=None,
        num_heads=None,
        num_encoder_layers=None,
        num_decoder_layers=None,
        dropout_rate=None,
        stop_grad_through_time=None,
        init_dist=None,
        init_seed=None,
        init_particles=None,
    ):
        """Run differentiable PF with optional runtime hyperparameter overrides."""
        if any(
            v is not None
            for v in (
                num_particles,
                ess_threshold,
                resample,
                proposal,
                d_model,
                hidden,
                num_heads,
                num_encoder_layers,
                num_decoder_layers,
                dropout_rate,
                stop_grad_through_time,
            )
        ):
            self.update_params(
                num_particles=num_particles,
                ess_threshold=ess_threshold,
                resample=resample,
                proposal=proposal,
                d_model=d_model,
                hidden=hidden,
                num_heads=num_heads,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                dropout_rate=dropout_rate,
                stop_grad_through_time=stop_grad_through_time,
            )
        return super().filter(
            y,
            num_particles=None,
            ess_threshold=None,
            resample=None,
            proposal=None,
            init_dist=init_dist,
            init_seed=init_seed,
            init_particles=init_particles,
        )


__all__ = [
    "ParticleTransformerResampler",
    "ParticleTransformerDPF",
]
