/**
 * sequence_layer.ts – Common interface implemented by all block types.
 *
 * Mamba1Block, Mamba2Block, Mamba3Block, and AttentionBlock all implement
 * SequenceLayer so that HybridMambaModel can iterate layers generically.
 */

export interface LayerForwardResult {
    output : GPUBuffer;
    cache  : unknown;  // type-specific per layer variant
}

export interface LayerParam {
    buf   : GPUBuffer;
    numel : number;
    name  : string;
}

export type LayerType = 'mamba1' | 'mamba2' | 'mamba3' | 'attention';

export interface SequenceLayer {
    readonly layerType: LayerType;

    forward(xBuf: GPUBuffer, batch: number, seqLen: number): LayerForwardResult;
    parameters(): LayerParam[];
    getTrainableParams(): LayerParam[];
    setWSLAMode(enabled: boolean): void;
    destroy(): void;
}
