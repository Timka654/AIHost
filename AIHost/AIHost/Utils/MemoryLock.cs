using System.Runtime.InteropServices;

namespace AIHost.Utils;

/// <summary>
/// Platform-specific memory locking to prevent swapping to disk
/// </summary>
public static class MemoryLock
{
    // Windows API
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool VirtualLock(IntPtr lpAddress, UIntPtr dwSize);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool VirtualUnlock(IntPtr lpAddress, UIntPtr dwSize);

    // Linux/POSIX API
    [DllImport("libc", SetLastError = true)]
    private static extern int mlock(IntPtr addr, UIntPtr len);

    [DllImport("libc", SetLastError = true)]
    private static extern int munlock(IntPtr addr, UIntPtr len);

    /// <summary>
    /// Lock memory region to prevent swapping
    /// </summary>
    public static bool Lock(IntPtr address, ulong size)
    {
        if (address == IntPtr.Zero || size == 0)
            return false;

        try
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                return VirtualLock(address, new UIntPtr(size));
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux) || 
                     RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            {
                return mlock(address, new UIntPtr(size)) == 0;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"⚠ Failed to lock memory: {ex.Message}");
            return false;
        }

        return false;
    }

    /// <summary>
    /// Unlock memory region
    /// </summary>
    public static bool Unlock(IntPtr address, ulong size)
    {
        if (address == IntPtr.Zero || size == 0)
            return false;

        try
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                return VirtualUnlock(address, new UIntPtr(size));
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux) || 
                     RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            {
                return munlock(address, new UIntPtr(size)) == 0;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"⚠ Failed to unlock memory: {ex.Message}");
            return false;
        }

        return false;
    }

    /// <summary>
    /// Check if memory locking is supported and available
    /// </summary>
    public static bool IsSupported()
    {
        return RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ||
               RuntimeInformation.IsOSPlatform(OSPlatform.Linux) ||
               RuntimeInformation.IsOSPlatform(OSPlatform.OSX);
    }

    /// <summary>
    /// Get recommended memory lock limit for the current platform
    /// Note: On Linux, check ulimit -l for the lock limit
    /// </summary>
    public static string GetRecommendations()
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            return "Windows: Requires 'Lock pages in memory' privilege (secpol.msc -> Local Policies -> User Rights Assignment)";
        }
        else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            return "Linux: Check 'ulimit -l' for memory lock limit. May require root or RLIMIT_MEMLOCK increase.";
        }
        else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            return "macOS: Memory locking available but may require elevated privileges.";
        }

        return "Memory locking not supported on this platform.";
    }
}
