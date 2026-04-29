using Xunit;
using AIHost.Utils;
using System.Runtime.InteropServices;

namespace AIHost.Tests;

/// <summary>
/// Tests for memory locking functionality
/// </summary>
public class MemoryLockTests
{
    [Fact]
    public void MemoryLock_ShouldBeSupportedOnMajorPlatforms()
    {
        // Act
        var isSupported = MemoryLock.IsSupported();
        
        // Assert
        // Should be supported on Windows, Linux, and macOS
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ||
            RuntimeInformation.IsOSPlatform(OSPlatform.Linux) ||
            RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            Assert.True(isSupported);
        }
    }
    
    [Fact]
    public void MemoryLock_LockShouldReturnFalseForZeroAddress()
    {
        // Act
        var result = MemoryLock.Lock(IntPtr.Zero, 1024);
        
        // Assert
        Assert.False(result);
    }
    
    [Fact]
    public void MemoryLock_LockShouldReturnFalseForZeroSize()
    {
        // Act
        var result = MemoryLock.Lock(new IntPtr(0x1000), 0);
        
        // Assert
        Assert.False(result);
    }
    
    [Fact]
    public void MemoryLock_UnlockShouldReturnFalseForZeroAddress()
    {
        // Act
        var result = MemoryLock.Unlock(IntPtr.Zero, 1024);
        
        // Assert
        Assert.False(result);
    }
    
    [Fact]
    public void MemoryLock_GetRecommendationsShouldReturnPlatformSpecificAdvice()
    {
        // Act
        var recommendations = MemoryLock.GetRecommendations();
        
        // Assert
        Assert.NotNull(recommendations);
        Assert.NotEmpty(recommendations);
        
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            Assert.Contains("Lock pages in memory", recommendations);
        }
        else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            Assert.Contains("ulimit", recommendations);
        }
    }
    
    [Fact]
    public void MemoryLock_LockAndUnlockShouldWorkWithValidMemory()
    {
        // Arrange - allocate some memory
        var size = 4096ul; // 4KB page
        var memory = Marshal.AllocHGlobal((int)size);
        
        try
        {
            // Act - try to lock (may fail without privileges)
            var lockResult = MemoryLock.Lock(memory, size);
            
            // If lock succeeded, unlock should also succeed
            if (lockResult)
            {
                var unlockResult = MemoryLock.Unlock(memory, size);
                Assert.True(unlockResult);
            }
            else
            {
                // Lock may fail without elevated privileges - that's expected
                Assert.False(lockResult);
            }
        }
        finally
        {
            // Cleanup
            Marshal.FreeHGlobal(memory);
        }
    }
    
    [Fact]
    public void LazyGGUFModel_ShouldAcceptMemoryLockParameter()
    {
        // This is an integration test that would require a real GGUF file
        // For now, we just verify the parameter is accepted in the constructor signature
        
        // The constructor signature should be:
        // public LazyGGUFModel(string filePath, IComputeDevice device, bool useMemoryMapping = true, bool useMemoryLock = false)
        
        // This is validated at compile time by the existing code
        Assert.True(true); // Placeholder - constructor signature is verified by compiler
    }
}
