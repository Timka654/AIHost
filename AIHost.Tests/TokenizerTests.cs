using AIHost.Tokenizer;

namespace AIHost.Tests;

public class TokenizerTests
{
    [Fact]
    public void BPETokenizer_Encode_ProducesValidTokens()
    {
        // Arrange
        var tokens = CreateDemoTokenizer();
        var tokenizer = new BPETokenizer(tokens, bosToken: 257, eosToken: 258, unknownToken: 256);

        // Act
        var result = tokenizer.Encode("Hello", addBos: true, addEos: false);

        // Assert
        Assert.NotEmpty(result);
        Assert.Equal(257, result[0]); // BOS token
    }

    [Fact]
    public void BPETokenizer_Decode_ReconstructsText()
    {
        // Arrange
        var tokens = CreateDemoTokenizer();
        var tokenizer = new BPETokenizer(tokens, bosToken: 257, eosToken: 258, unknownToken: 256);
        
        var encoded = tokenizer.Encode("Hello", addBos: false, addEos: false);

        // Act
        var decoded = tokenizer.Decode(encoded);

        // Assert - demo tokenizer may add unknown tokens for missing characters
        Assert.Contains("Hello", decoded);
    }

    [Fact]
    public void BPETokenizer_GetToken_ReturnsCorrectToken()
    {
        // Arrange
        var tokens = CreateDemoTokenizer();
        var tokenizer = new BPETokenizer(tokens, bosToken: 257, eosToken: 258, unknownToken: 256);

        // Act
        var bosToken = tokenizer.GetToken(257);
        var eosToken = tokenizer.GetToken(258);

        // Assert
        Assert.Equal("<s>", bosToken);
        Assert.Equal("</s>", eosToken);
    }

    /// <summary>
    /// Verifies that special tokens in chat templates (e.g. <|system|>, <|user|>)
    /// are NOT corrupted by SentencePiece ▁ normalisation.
    /// Regression test for the bug where "▁<|system|>" was produced instead of "<|system|>".
    /// </summary>
    [Fact]
    public void BPETokenizer_Encode_ChatTemplateSpecialTokens_ArePreserved()
    {
        // Arrange — tokenizer that includes TinyLlama-style special tokens
        var tokens = new List<string>();
        for (int i = 0; i < 256; i++) tokens.Add(((char)i).ToString());
        tokens.Add("<unk>");       // 256
        tokens.Add("<s>");         // 257 (BOS)
        tokens.Add("</s>");        // 258 (EOS)
        tokens.Add("<|system|>");  // 259
        tokens.Add("<|user|>");    // 260
        tokens.Add("<|assistant|>"); // 261
        tokens.Add("▁");           // 262 (SentencePiece space)
        tokens.Add("You");         // 263
        tokens.Add("are");         // 264
        tokens.Add("a");           // 265
        tokens.Add("helpful");     // 266
        tokens.Add("AI");          // 267
        tokens.Add("assistant");   // 268
        tokens.Add("Who");         // 269
        tokens.Add("?");           // 270

        var tokenizer = new BPETokenizer(tokens.ToArray(), bosToken: 257, eosToken: 258, unknownToken: 256);

        // Act — encode a TinyLlama-style chat template prompt
        string prompt = "<|system|>\nYou are a helpful AI assistant.\n</s>\n<|user|>\nWho are you?\n</s>\n<|assistant|>\n";
        var result = tokenizer.Encode(prompt, addBos: false, addEos: false);

        // Assert — special tokens must be present as exact IDs, not <unk>
        int unkId = 256;
        Assert.DoesNotContain(unkId, result); // No <unk> tokens!

        // Verify special tokens appear at expected positions
        Assert.Contains(259, result); // <|system|>
        Assert.Contains(260, result); // <|user|>
        Assert.Contains(261, result); // <|assistant|>
        Assert.Contains(258, result); // </s>

        // Verify the sequence starts with <|system|>
        Assert.Equal(259, result[0]);
    }

    /// <summary>
    /// Verifies that Qwen-style special tokens (<|im_start|>, <|im_end|>)
    /// are also preserved correctly.
    /// </summary>
    [Fact]
    public void BPETokenizer_Encode_QwenSpecialTokens_ArePreserved()
    {
        // Arrange
        var tokens = new List<string>();
        for (int i = 0; i < 256; i++) tokens.Add(((char)i).ToString());
        tokens.Add("<unk>");         // 256
        tokens.Add("<s>");           // 257 (BOS)
        tokens.Add("</s>");          // 258 (EOS)
        tokens.Add("<|im_start|>");  // 259
        tokens.Add("<|im_end|>");    // 260
        tokens.Add("▁");             // 261
        tokens.Add("Hello");         // 262

        var tokenizer = new BPETokenizer(tokens.ToArray(), bosToken: 257, eosToken: 258, unknownToken: 256);

        // Act
        string prompt = "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>assistant\n";
        var result = tokenizer.Encode(prompt, addBos: false, addEos: false);

        // Assert
        int unkId = 256;
        Assert.DoesNotContain(unkId, result);
        Assert.Contains(259, result); // <|im_start|>
        Assert.Contains(260, result); // <|im_end|>
        Assert.Equal(259, result[0]); // starts with <|im_start|>
    }

    private static string[] CreateDemoTokenizer()
    {
        var tokens = new List<string>();
        
        // Byte tokens
        for (int i = 0; i < 256; i++)
        {
            tokens.Add(((char)i).ToString());
        }
        
        // Special tokens
        tokens.Add("<unk>");  // 256
        tokens.Add("<s>");    // 257 (BOS)
        tokens.Add("</s>");   // 258 (EOS)
        
        // Common words
        string[] words = { " ", "Hello", "world", "test", "!" };
        tokens.AddRange(words);
        
        return tokens.ToArray();
    }
}
