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

        // Assert
        Assert.Equal("Hello", decoded);
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
