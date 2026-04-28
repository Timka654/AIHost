namespace AIHost;

/// <summary>
/// Entry point - routes between web server and manual tests
/// </summary>
internal class Program
{
    static async Task Main(string[] args)
    {
        // Check for --web flag to start web server
        if (args.Contains("--web"))
        {
            await WebServer.RunAsync(args);
            return;
        }

        // Otherwise run manual tests
        ManualTests.TestRunner.RunInteractiveTests(args);
    }
}
