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
            // Web server entry is in WebServer.cs (top-level statements).
            // This branch is unreachable when that file is the program entry point.
            Console.WriteLine("Use 'dotnet run' without --manual to start the web server.");
            return;
        }

        // Otherwise run manual tests
        ManualTests.TestRunner.RunInteractiveTests(args);
    }
}
