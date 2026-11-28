import { SessionError, type SessionErrorCode } from './errors.js';

export type SessionAssetKind = 'checkpoint' | 'tokenizer-vocab' | 'tokenizer-merges';

export interface SessionAssetPolicy {
    /** Read a previously installed asset. Return undefined when it is absent. */
    readLocal(kind: SessionAssetKind, url: string): Promise<Response | undefined>;
    /** Evaluated on each cache miss, including retry and subsequent checkpoint loads. */
    allowsNetwork(kind: SessionAssetKind): boolean;
}

export async function resolveSessionAsset(
    kind: SessionAssetKind,
    url: string,
    policy?: SessionAssetPolicy,
): Promise<Response> {
    const local = await policy?.readLocal(kind, url);
    if (!local && policy && !policy.allowsNetwork(kind)) {
        throw new SessionError('ASSET_UNAVAILABLE_OFFLINE', `No local ${kind} asset is installed.`);
    }
    const response = local ?? await fetch(url);
    if (!response.ok) {
        const code: SessionErrorCode = kind === 'checkpoint' ? 'CHECKPOINT_FETCH_FAILED' : 'TOKENIZER_LOAD_FAILED';
        throw new SessionError(code, `HTTP ${response.status} ${response.statusText}`);
    }
    return response;
}
