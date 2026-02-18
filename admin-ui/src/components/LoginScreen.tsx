/* ============================================================================
 * Login Screen
 * ============================================================================ */

import { useState, type FormEvent } from 'react';
import { useApp } from '../store';

export default function LoginScreen() {
  const { login } = useApp();
  const [key, setKey] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!key.trim()) {
      setError('Please enter the admin API key');
      return;
    }
    setLoading(true);
    setError('');
    try {
      await login(key.trim());
    } catch (err) {
      setError('Login failed: ' + (err instanceof Error ? err.message : 'Unknown error'));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-80 flex items-center justify-center z-50">
      <form onSubmit={handleSubmit} className="bg-white p-8 rounded-lg shadow-xl max-w-md w-full mx-4">
        <h2 className="text-2xl font-bold text-center mb-6 text-blue-600">Qdrant Proxy Admin</h2>
        <p className="text-gray-600 text-center mb-6">Enter your admin API key to access the management interface.</p>

        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">Admin API Key</label>
          <input
            type="password"
            value={key}
            onChange={(e) => setKey(e.target.value)}
            className="w-full px-4 py-2 border-2 border-gray-300 rounded-md focus:border-blue-500 focus:outline-none"
            placeholder="Enter QDRANT_PROXY_ADMIN_KEY"
            autoFocus
          />
        </div>

        <button type="submit" disabled={loading} className="btn-primary w-full py-3">
          {loading ? 'Logging in...' : 'Login'}
        </button>

        {error && <div className="mt-4 text-red-600 text-sm text-center">{error}</div>}
      </form>
    </div>
  );
}
