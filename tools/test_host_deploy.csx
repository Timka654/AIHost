#r "nuget: SSH.NET, 2025.1.0"

using System.Text;
using System.Text.Encodings;
using Renci.SshNet;
using System;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using System.Net.Http;
using System.Runtime.CompilerServices;


string GetCurrentFilePath([CallerFilePath] string path = "") => path;

var currentDir = Path.GetDirectoryName(GetCurrentFilePath());

var solutionRoot = Path.GetFullPath(Path.Combine(currentDir, "../")).TrimEnd('\\', '/');

// ==========================================
// 1. КОНФИГУРАЦИЯ И ПУТИ
// ==========================================
var projectName = "AIHost";
var registry = "doreg.mtvworld.net"; // Замени на свой адрес registry, или оставь пустым для Docker Hub
var tag = "latest";
var imageName = string.IsNullOrEmpty(registry) ? projectName.ToLower() : $"{registry}/{projectName.ToLower()}";

// Скрипт запускается из папки <КореньРешения>/tools 
var projectPath = Path.Combine(solutionRoot, projectName, $"{projectName}.csproj");

// Папка для артефактов публикации (если Dockerfile не использует многоэтапную сборку)
var publishDir = Path.Combine(solutionRoot, "publish_output");

// Для .NET проектов контекстом сборки Docker обычно выступает корень решения
var dockerContext = solutionRoot;
var dockerfilePath = Path.Combine(solutionRoot, projectName, "Dockerfile");

Console.WriteLine($"[INFO] Подготовка к публикации: {imageName}:{tag}");
Console.WriteLine($"[INFO] Корень решения: {solutionRoot}");

// ==========================================
// 2. ОСНОВНОЙ ПРОЦЕСС
// ==========================================
try
{
    // Шаг 1: Авторизация (пароль подтянется из хранилища)
    Console.WriteLine("\n[1/4] Авторизация в Docker Registry...");
    var loginArgs = string.IsNullOrEmpty(registry) ? "login" : $"login {registry}";
    await RunProcessAsync("docker", loginArgs);

    // Шаг 2: Сборка .NET проекта
    // (Если в твоем Dockerfile используется многоэтапная сборка (multi-stage build), 
    // этот шаг можно закомментировать, так как dotnet publish произойдет внутри контейнера)
    Console.WriteLine("\n[2/4] Локальная сборка .NET проекта...");
    await RunProcessAsync("dotnet", $"publish \"{projectPath}\" -c Release -o \"{publishDir}\"");

    // Шаг 3: Сборка Docker-образа
    Console.WriteLine("\n[3/4] Сборка Docker-образа...");
    await RunProcessAsync("docker", $"build -t {imageName}:{tag} -f \"{dockerfilePath}\" \"{dockerContext}\"");

    // Шаг 4: Пуш в реестр
    Console.WriteLine("\n[4/4] Отправка образа в Registry...");
    await RunProcessAsync("docker", $"push {imageName}:{tag}");

    Console.WriteLine("\n[SUCCESS] Публикация успешно завершена!");
}
catch (Exception ex)
{
    Console.Error.WriteLine($"\n[ERROR] Ошибка в процессе публикации: {ex.Message}");
    Environment.Exit(1);
}

// ==========================================
// 3. ИСПОЛНЕНИЕ КОМАНД
// ==========================================
async Task RunProcessAsync(string command, string arguments)
{
    var psi = new ProcessStartInfo
    {
        FileName = command,
        Arguments = arguments,
        RedirectStandardOutput = true,
        RedirectStandardError = true,
        UseShellExecute = false,
        CreateNoWindow = true
    };

    using var process = new Process { StartInfo = psi };

    process.OutputDataReceived += (sender, e) => { if (e.Data != null) Console.WriteLine(e.Data); };
    process.ErrorDataReceived += (sender, e) => { if (e.Data != null) Console.Error.WriteLine(e.Data); };

    process.Start();
    process.BeginOutputReadLine();
    process.BeginErrorReadLine();

    await process.WaitForExitAsync();

    if (process.ExitCode != 0)
    {
        throw new Exception($"Команда '{command}' завершилась с кодом {process.ExitCode}");
    }
}




// ... (твой код сборки и публикации образа) ...

Console.WriteLine("\n[INFO] Подключение к TrueNAS для обновления стека...");

var host = "192.168.88.201"; // IP твоего TrueNAS
var username = "root"; // Пользователь, которому ты добавил ключ в TrueNAS

// Путь к приватному ключу. 
// Если ключ лежит в папке tools рядом с Program.cs:
var keyFilePath = Path.GetFullPath(@"D:\User\Downloads\deploy_key");
var stackPath = "/mnt/.ix-apps/app_mounts/dockge/stacks/ai_host";

try
{
    var keyFile = new PrivateKeyFile(keyFilePath);
    using (var client = new SshClient(host, username, new[] { keyFile }))
    {
        client.Connect();
        Console.WriteLine("[INFO] SSH соединение с TrueNAS установлено.");

        // Формируем команду обновления
        var commandText = $"cd {stackPath} && docker compose pull && docker compose up -d";

        Console.WriteLine($"[INFO] Выполнение: {commandText}");
        var cmd = client.CreateCommand(commandText);
        var result = cmd.Execute();

        Console.WriteLine(result);

        if (!string.IsNullOrEmpty(cmd.Error))
        {
            // Часто docker compose пишет прогресс pull в stderr, поэтому это не всегда критичная ошибка, 
            // но стоит вывести ее в консоль для диагностики
            Console.WriteLine($"[SSH STDERR]: {cmd.Error}");
        }

        client.Disconnect();
        Console.WriteLine("[INFO] Команда выполнена. Проверь статус в Dockge!");
    }
}
catch (Exception ex)
{
    Console.Error.WriteLine($"[ERROR] Ошибка SSH-подключения: {ex.Message}");
}






using (var http = new HttpClient())
{


    var request = new HttpRequestMessage(HttpMethod.Get, $"http://192.168.88.201:11434/manage/build_version");

    var response = await http.SendAsync(request);

    var json = await response.Content.ReadAsStringAsync();

    Console.WriteLine($"build version: {json}");
}