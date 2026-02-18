/* ============================================================================
 * Shared UI primitives: Modal, Spinner
 * ============================================================================ */

import type { ReactNode } from 'react';

/** Full-screen modal overlay with centered content card. */
export function Modal({
  open,
  onClose,
  title,
  children,
  maxWidth = 'max-w-2xl',
}: {
  open: boolean;
  onClose: () => void;
  title: string;
  children: ReactNode;
  maxWidth?: string;
}) {
  if (!open) return null;
  return (
    <div
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div className={`bg-white rounded-lg ${maxWidth} w-full shadow-xl max-h-[90vh] overflow-y-auto`}>
        <div className="p-6">
          <div className="flex justify-between items-center mb-6">
            <h3 className="text-xl font-bold text-gray-800">{title}</h3>
            <button onClick={onClose} className="text-gray-400 hover:text-gray-600 text-2xl">
              &times;
            </button>
          </div>
          {children}
        </div>
      </div>
    </div>
  );
}

/** Inline loading spinner with optional label. */
export function Spinner({ label, className }: { label?: string; className?: string }) {
  return (
    <div className={`text-center py-8 ${className || ''}`}>
      <div className="loading-spinner mx-auto mb-2" />
      {label && <p className="text-sm text-gray-500">{label}</p>}
    </div>
  );
}

/** FAQ entry display: Question → Answer */
export function FAQEntryDisplay({
  question,
  answer,
}: {
  question: string;
  answer: string;
}) {
  return (
    <div className="flex flex-col gap-1">
      <div className="font-medium text-gray-700">Q: {question}</div>
      <div className="text-gray-600">A: {answer}</div>
    </div>
  );
}

/** Star rating buttons (1–5). */
export function StarRating({
  onRate,
  disabled,
}: {
  onRate: (score: number) => void;
  disabled: boolean;
}) {
  return (
    <>
      {[1, 2, 3, 4, 5].map((s) => (
        <button
          key={s}
          disabled={disabled}
          onClick={() => onRate(s)}
          className="px-1.5 py-0.5 rounded text-xs bg-yellow-50 hover:bg-yellow-200 text-yellow-700 border border-yellow-200 disabled:opacity-50"
          title={`Rank ${s}/5`}
        >
          {'★'.repeat(s)}
        </button>
      ))}
    </>
  );
}
