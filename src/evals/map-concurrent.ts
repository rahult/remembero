/**
 * A fixed pool of workers over a list, where the callback is told which worker it is
 * running on. The worker id matters when a run holds one resource per worker — an adapter
 * process, a connection — because selecting that resource by the item's position instead
 * lets two in-flight items land on the same one while another sits idle.
 */
export async function mapConcurrent<T, R>(
  values: readonly T[],
  concurrency: number,
  operation: (value: T, index: number, workerId: number) => Promise<R>,
): Promise<R[]> {
  const results = new Array<R>(values.length);
  let next = 0;
  await Promise.all(
    Array.from(
      { length: Math.min(concurrency, values.length) },
      async (_unused, workerId) => {
        while (true) {
          const index = next++;
          if (index >= values.length) return;
          results[index] = await operation(values[index]!, index, workerId);
        }
      },
    ),
  );
  return results;
}
