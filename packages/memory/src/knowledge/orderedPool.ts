/**
 * knowledge/orderedPool.ts — an ORDER-PRESERVING bounded-concurrency prefetcher.
 *
 * `prefetchOrdered` starts up to `limit` jobs ahead of the consumer and yields their settled
 * results strictly in input order, so a consumer that acts on each result in sequence behaves
 * exactly as if the jobs had run one at a time — only the waiting overlaps. Jobs start only when
 * the consumer pulls (`next()`), so `limit` 1 has no read-ahead at all. A rejected job is
 * yielded as `rejected`, never thrown. Nothing new starts once `signal` aborts or the consumer
 * stops early. No timers, no randomness.
 */

export interface Prefetched<T, R> {
    item: T;
    i: number;
    result: PromiseSettledResult<R>;
}

export async function* prefetchOrdered<T, R>(
    items: readonly T[],
    limit: number,
    work: (item: T, i: number) => Promise<R>,
    signal?: AbortSignal,
): AsyncGenerator<Prefetched<T, R>> {
    const width = Math.max(1, Math.floor(limit));
    const pending: Array<Promise<PromiseSettledResult<R>>> = [];
    let next = 0;
    const fill = () => {
        while (pending.length < width && next < items.length && !signal?.aborted) {
            const i = next++;
            pending.push(
                (async () => work(items[i]!, i))().then(
                    (value): PromiseSettledResult<R> => ({ status: 'fulfilled', value }),
                    (reason): PromiseSettledResult<R> => ({ status: 'rejected', reason }),
                ),
            );
        }
    };
    try {
        for (let i = 0; i < items.length; i++) {
            // The consumer's turn for item i begins here: an abort by now ends the stream, discarding read-ahead.
            if (signal?.aborted) return;
            fill();
            const head = pending.shift();
            if (!head) return; // aborted before this item started
            const result = await head;
            yield { item: items[i]!, i, result };
        }
    } finally {
        // Never complete while a started job is still running: nothing outlives the consumer.
        await Promise.allSettled(pending);
    }
}
