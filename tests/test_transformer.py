import pytest
import numpy as np
import tensorflow as tf
from src.models.transformer import (
    AttentionMatrix,
    AttentionHead,
    MultiHeadAttention,
    PositionalEncoding,
    LanguageTransformerBlock,
    TransformerLanguageModel
)


class TestAttentionMatrix:
    """Tests for AttentionMatrix layer"""

    @pytest.fixture
    def attention_layer(self):
        """Create attention matrix layer"""
        return AttentionMatrix()

    def test_initialization(self, attention_layer):
        """Test that AttentionMatrix can be instantiated"""
        assert attention_layer is not None, "AttentionMatrix should instantiate"

    def test_forward_pass(self, attention_layer):
        """Test that attention matrix can compute attention weights"""
        batch_size = 2
        seq_length = 5
        embed_size = 8

        K = tf.random.normal([batch_size, seq_length, embed_size])
        Q = tf.random.normal([batch_size, seq_length, embed_size])

        try:
            weights = attention_layer([K, Q])
            assert weights is not None, "Should return attention weights"
        except NotImplementedError:
            pytest.fail("AttentionMatrix raises NotImplementedError")

    def test_output_shape(self, attention_layer):
        """Test that output has correct shape"""
        batch_size = 2
        seq_length = 5
        embed_size = 8

        K = tf.random.normal([batch_size, seq_length, embed_size])
        Q = tf.random.normal([batch_size, seq_length, embed_size])

        try:
            weights = attention_layer([K, Q])
            # Output should be [batch_size, seq_length, seq_length]
            expected_shape = (batch_size, seq_length, seq_length)
            assert weights.shape == expected_shape, \
                f"Attention weights shape should be {expected_shape}, got {weights.shape}"
        except NotImplementedError:
            pytest.skip("AttentionMatrix not implemented")


class TestAttentionHead:
    """Tests for AttentionHead layer"""

    @pytest.fixture
    def attention_head(self):
        """Create single attention head"""
        return AttentionHead(input_size=64, output_size=32)

    def test_initialization(self, attention_head):
        """Test that AttentionHead can be instantiated"""
        assert attention_head is not None, "AttentionHead should instantiate"

    def test_forward_pass(self, attention_head):
        """Test that attention head can process inputs"""
        batch_size = 2
        seq_length = 5
        input_size = 64

        inputs = tf.random.normal([batch_size, seq_length, input_size])

        try:
            output = attention_head(inputs, inputs, inputs)
            assert output is not None, "Should return output"
        except NotImplementedError:
            pytest.fail("AttentionHead raises NotImplementedError")

    def test_output_shape(self, attention_head):
        """Test that output has correct shape"""
        batch_size = 2
        seq_length = 5
        input_size = 64
        output_size = 32

        inputs = tf.random.normal([batch_size, seq_length, input_size])

        try:
            output = attention_head(inputs, inputs, inputs)
            expected_shape = (batch_size, seq_length, output_size)
            assert output.shape == expected_shape, \
                f"Output shape should be {expected_shape}, got {output.shape}"
        except NotImplementedError:
            pytest.skip("AttentionHead not implemented")


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention layer"""

    @pytest.fixture
    def multi_head_attention(self):
        """Create multi-head attention layer"""
        return MultiHeadAttention(embed_size=64, num_heads=8)

    def test_initialization(self, multi_head_attention):
        """Test that MultiHeadAttention can be instantiated"""
        assert multi_head_attention is not None, "MultiHeadAttention should instantiate"
        assert multi_head_attention.num_heads == 8, "Should have correct number of heads"

    def test_embed_size_divisible_by_heads(self):
        """Test that embed_size must be divisible by num_heads"""
        with pytest.raises(AssertionError):
            MultiHeadAttention(embed_size=65, num_heads=8)

    def test_forward_pass(self, multi_head_attention):
        """Test that multi-head attention can process inputs"""
        batch_size = 2
        seq_length = 5
        embed_size = 64

        inputs = tf.random.normal([batch_size, seq_length, embed_size])

        try:
            output = multi_head_attention(inputs, inputs, inputs)
            assert output is not None, "Should return output"
        except NotImplementedError:
            pytest.fail("MultiHeadAttention raises NotImplementedError")

    def test_output_shape(self, multi_head_attention):
        """Test that output has correct shape"""
        batch_size = 2
        seq_length = 5
        embed_size = 64

        inputs = tf.random.normal([batch_size, seq_length, embed_size])

        try:
            output = multi_head_attention(inputs, inputs, inputs)
            expected_shape = (batch_size, seq_length, embed_size)
            assert output.shape == expected_shape, \
                f"Output shape should be {expected_shape}, got {output.shape}"
        except NotImplementedError:
            pytest.skip("MultiHeadAttention not implemented")


class TestPositionalEncoding:
    """Tests for PositionalEncoding layer"""

    @pytest.fixture
    def positional_encoding(self):
        """Create positional encoding layer"""
        return PositionalEncoding(d_model=64, max_seq_length=100)

    def test_forward_pass(self, positional_encoding):
        """Test that positional encodings can be added to inputs"""
        batch_size = 2
        seq_length = 10
        d_model = 64

        inputs = tf.random.normal([batch_size, seq_length, d_model])

        try:
            output = positional_encoding(inputs)
            assert output is not None, "Should return output with positional encodings"
        except NotImplementedError:
            pytest.fail("PositionalEncoding forward pass raises NotImplementedError")

    def test_output_shape_preserved(self, positional_encoding):
        """Test that output has same shape as input"""
        batch_size = 2
        seq_length = 10
        d_model = 64

        inputs = tf.random.normal([batch_size, seq_length, d_model])

        try:
            output = positional_encoding(inputs)
            assert output.shape == inputs.shape, \
                "Output shape should match input shape"
        except NotImplementedError:
            pytest.skip("PositionalEncoding not implemented")


class TestLanguageTransformerBlock:
    """Tests for LanguageTransformerBlock layer"""

    @pytest.fixture
    def transformer_block(self):
        """Create transformer block"""
        return LanguageTransformerBlock(embed_size=64, num_heads=8)

    def test_initialization(self, transformer_block):
        """Test that transformer block can be instantiated"""
        assert transformer_block is not None, "LanguageTransformerBlock should instantiate"

    def test_forward_pass(self, transformer_block):
        """Test that transformer block can process inputs"""
        batch_size = 2
        seq_length = 5
        embed_size = 64

        inputs = tf.random.normal([batch_size, seq_length, embed_size])

        try:
            output = transformer_block(inputs, training=False)
            assert output is not None, "Should return output"
        except NotImplementedError:
            pytest.fail("LanguageTransformerBlock raises NotImplementedError")

    def test_output_shape(self, transformer_block):
        """Test that output has correct shape"""
        batch_size = 2
        seq_length = 5
        embed_size = 64

        inputs = tf.random.normal([batch_size, seq_length, embed_size])

        try:
            output = transformer_block(inputs, training=False)
            assert output.shape == inputs.shape, \
                "Output shape should match input shape"
        except NotImplementedError:
            pytest.skip("LanguageTransformerBlock not implemented")


class TestTransformerLanguageModel:
    """Tests for complete TransformerLanguageModel"""

    @pytest.fixture
    def small_transformer(self):
        """Create small transformer model for testing"""
        return TransformerLanguageModel(
            vocab_size=100,
            d_model=64,
            n_heads=4,
            n_layers=2,
            max_seq_length=50,
            dropout_rate=0.1
        )

    def test_initialization(self, small_transformer):
        """Test that transformer model can be instantiated"""
        assert small_transformer is not None, "TransformerLanguageModel should instantiate"
        assert small_transformer.vocab_size == 100, "Should store vocab_size"
        assert small_transformer.d_model == 64, "Should store d_model"

    def test_forward_pass(self, small_transformer):
        """Test that transformer can process input sequences"""
        batch_size = 2
        seq_length = 10
        vocab_size = 100

        inputs = tf.random.uniform([batch_size, seq_length], maxval=vocab_size, dtype=tf.int32)

        try:
            outputs = small_transformer(inputs, training=False)
            assert outputs is not None, "Should return output logits"
        except NotImplementedError:
            pytest.fail("TransformerLanguageModel raises NotImplementedError")

    def test_output_shape(self, small_transformer):
        """Test that output has correct shape"""
        batch_size = 2
        seq_length = 10
        vocab_size = 100

        inputs = tf.random.uniform([batch_size, seq_length], maxval=vocab_size, dtype=tf.int32)

        try:
            outputs = small_transformer(inputs, training=False)
            # Output should be [batch_size, seq_length, vocab_size]
            expected_shape = (batch_size, seq_length, vocab_size)
            assert outputs.shape == expected_shape, \
                f"Output shape should be {expected_shape}, got {outputs.shape}"
        except NotImplementedError:
            pytest.skip("TransformerLanguageModel not implemented")

    def test_output_dtype(self, small_transformer):
        """Test that output has correct dtype"""
        batch_size = 2
        seq_length = 5
        vocab_size = 100

        inputs = tf.random.uniform([batch_size, seq_length], maxval=vocab_size, dtype=tf.int32)

        try:
            outputs = small_transformer(inputs, training=False)
            assert outputs.dtype == tf.float32, "Output should be float32"
        except NotImplementedError:
            pytest.skip("TransformerLanguageModel not implemented")

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
