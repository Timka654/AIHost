using Xunit;

// GPU tests must not run in parallel — Vulkan command queues are not thread-safe.
// DisableTestParallelization prevents concurrent execution across test classes.
[assembly: CollectionBehavior(DisableTestParallelization = true, MaxParallelThreads = 1)]
