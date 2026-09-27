/**
 * experience/approval.ts — which learned steps pause for the person: the ones that
 * cannot be taken back. A click on "Send", "Pay", "Delete" or "Submit" is where a replay
 * that drifted does real damage, so the gate is on by default there. Review can still
 * turn it on or off per step.
 */
import type { Action } from './types.js';

/** Words, in the five product languages, that name an action with consequences outside the app. */
const IRREVERSIBLE = new Set([
  // en
  'send', 'submit', 'pay', 'purchase', 'buy', 'order', 'checkout', 'delete', 'remove', 'confirm', 'transfer',
  'publish', 'post', 'sign', 'approve', 'book', 'cancel', 'erase', 'wipe', 'format', 'uninstall',
  // de
  'senden', 'absenden', 'bezahlen', 'kaufen', 'bestellen', 'löschen', 'entfernen', 'bestätigen', 'überweisen', 'veröffentlichen',
  // es
  'enviar', 'pagar', 'comprar', 'pedido', 'eliminar', 'borrar', 'confirmar', 'transferir', 'publicar',
  // fr
  'envoyer', 'payer', 'acheter', 'commander', 'supprimer', 'confirmer', 'virement', 'publier',
]);

/** CJK has no word boundaries, so these are matched as substrings. */
const IRREVERSIBLE_CJK = ['发送', '提交', '支付', '付款', '购买', '下单', '删除', '确认', '转账', '发布'];

export function isIrreversibleLabel(label: string): boolean {
  const lower = label.toLowerCase();
  if (IRREVERSIBLE_CJK.some((w) => lower.includes(w))) return true;
  return lower.split(/[^\p{L}\p{N}]+/u).some((w) => w.length > 0 && IRREVERSIBLE.has(w));
}

/** The label a person would use for an element: its name, else its automation id. */
export function elementLabel(target: { name: string; automationId: string } | null | undefined): string {
  if (!target) return '';
  return target.name.trim() || target.automationId.trim();
}

/** The default approval gate for a recorded action. */
export function needsApproval(action: Action): boolean {
  switch (action.kind) {
    case 'click':
      return isIrreversibleLabel(elementLabel(action.target));
    case 'keys':
      // Enter can submit the form it is pressed in, but which button fires is not in the
      // recording — so it is gated only when the element itself is labelled irreversible.
      return action.keys.toLowerCase() === 'enter' && isIrreversibleLabel(elementLabel(action.target));
    case 'setValue':
      return false;
  }
}
