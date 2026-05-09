using System.Text;
using System.Text.Encodings;
using System.Net.Http;

using (var http = new HttpClient())
{
        var request = new HttpRequestMessage(HttpMethod.Get, $"http://192.168.88.201:11434/manage/build_version");

        var response = await http.SendAsync(request);

        var json = await response.Content.ReadAsStringAsync();

        Console.WriteLine(json);
}