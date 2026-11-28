import { afterEach, describe, expect, it, jest } from '@jest/globals';
import { resolveSessionAsset } from '../src/session/assets.js';

const URL = 'https://assets.example/model.bin';

afterEach(() => jest.restoreAllMocks());

describe('session assets', () => {
    it('reads an installed asset without opening the network', async () => {
        const fetch = jest.spyOn(globalThis, 'fetch');
        const local = new Response('installed');
        const allowsNetwork = jest.fn(() => false);
        const result = await resolveSessionAsset('checkpoint', URL, {
            readLocal: async () => local,
            allowsNetwork,
        });
        expect(await result.text()).toBe('installed');
        expect(allowsNetwork).not.toHaveBeenCalled();
        expect(fetch).not.toHaveBeenCalled();
    });

    it('refuses a missing asset before fetching and checks permission on subsequent loads', async () => {
        const fetch = jest.spyOn(globalThis, 'fetch').mockResolvedValue(new Response('downloaded'));
        let online = false;
        const policy = {
            readLocal: async () => undefined,
            allowsNetwork: () => online,
        };
        await expect(resolveSessionAsset('tokenizer-vocab', URL, policy)).rejects.toMatchObject({
            code: 'ASSET_UNAVAILABLE_OFFLINE',
        });
        expect(fetch).not.toHaveBeenCalled();
        online = true;
        const result = await resolveSessionAsset('tokenizer-vocab', URL, policy);
        expect(await result.text()).toBe('downloaded');
        expect(fetch).toHaveBeenCalledWith(URL);
    });
});
