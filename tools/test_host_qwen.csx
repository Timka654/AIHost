using System.Text;
using System.Text.Encodings;
using System.Net.Http;

using (var http = new HttpClient())
{
    http.Timeout = TimeSpan.FromMinutes(10);

    var request = new HttpRequestMessage(HttpMethod.Post, $"http://192.168.88.201:11434/manage/chat");

    request.Content = new StringContent(
    "{\"model_name\":\"example-model\",\"message\":\"Hi\",\"system_message\":null,\"temperature\":0.7,\"max_tokens\":512,\"stream\":true}", Encoding.UTF8, "application/json");

    var response = await http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead);

    using var stream = await response.Content.ReadAsStreamAsync();

    using var reader = new StreamReader(stream);
    while (!reader.EndOfStream)
    {
        var line = await reader.ReadLineAsync();

        var json = line;
        Console.WriteLine(json);
        if (json.Contains("\"done\":true"))
            break;
    }
}