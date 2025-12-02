import { useMemo, useEffect, useCallback } from 'react';

import { useSetState } from 'src/hooks/use-set-state';

import axios, { endpoints } from 'src/utils/axios';

import { STORAGE_KEY } from './constant';
import { AuthContext } from '../auth-context';
import { setSession, isValidToken } from './utils';

// ----------------------------------------------------------------------

export function AuthProvider({ children }) {
  const { state, setState } = useSetState({
    user: null,
    loading: true,
  });

  const checkUserSession = useCallback(async () => {
    try {
      const accessToken = localStorage.getItem(STORAGE_KEY);

      if (accessToken && isValidToken(accessToken)) {
        setSession(accessToken);

        const res = await axios.get(endpoints.auth.me);

        const { user } = res.data;

        setState({ user: { ...user, accessToken }, loading: false });
      } else {
        setState({ user: null, loading: false });
      }
    } catch (error) {
      console.error(error);
      setState({ user: null, loading: false });
    }
  }, [setState]);

  const login = useCallback(async (email, password) => {
    try {
      const response = await axios.post(endpoints.auth.signIn, {
        email,
        password,
      });

      const { accessToken, user } = response.data;

      setSession(accessToken);

      setState({ user: { ...user, accessToken }, loading: false });

      // Notify other tabs about auth change
      try {
        const channel = new BroadcastChannel('auth_channel');
        channel.postMessage({ type: 'AUTH_CHANGE' });
        channel.close();
      } catch (error) {
        // BroadcastChannel not supported, ignore
      }

      return user;
    } catch (error) {
      console.error('Login error:', error);
      throw error;
    }
  }, [setState]);

  const logout = useCallback(() => {
    setSession(null);
    setState({ user: null, loading: false });

    // Notify other tabs about logout
    try {
      const channel = new BroadcastChannel('auth_channel');
      channel.postMessage({ type: 'LOGOUT' });
      channel.close();
    } catch (error) {
      // BroadcastChannel not supported, ignore
    }
  }, [setState]);

  useEffect(() => {
    checkUserSession();

    // Listen for storage changes (when token is updated in another tab)
    const handleStorageChange = (e) => {
      if (e.key === STORAGE_KEY) {
        if (e.newValue) {
          // Token was set in another tab
          checkUserSession();
        } else {
          // Token was removed in another tab (logout)
          setState({ user: null, loading: false });
        }
      }
    };

    // Use BroadcastChannel for cross-tab communication (more reliable)
    let broadcastChannel = null;
    try {
      broadcastChannel = new BroadcastChannel('auth_channel');
      broadcastChannel.onmessage = (event) => {
        if (event.data.type === 'AUTH_CHANGE') {
          checkUserSession();
        } else if (event.data.type === 'LOGOUT') {
          setState({ user: null, loading: false });
        }
      };
    } catch (error) {
      // BroadcastChannel not supported, fallback to storage event
      console.warn('BroadcastChannel not supported, using storage event');
    }

    window.addEventListener('storage', handleStorageChange);

    return () => {
      window.removeEventListener('storage', handleStorageChange);
      if (broadcastChannel) {
        broadcastChannel.close();
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ----------------------------------------------------------------------

  const checkAuthenticated = state.user ? 'authenticated' : 'unauthenticated';

  const status = state.loading ? 'loading' : checkAuthenticated;

  const memoizedValue = useMemo(
    () => ({
      user: state.user
        ? {
            ...state.user,
            role: state.user?.role ?? 'doctor',
          }
        : null,
      login,
      logout,
      checkUserSession,
      loading: status === 'loading',
      authenticated: status === 'authenticated',
      unauthenticated: status === 'unauthenticated',
    }),
    [checkUserSession, state.user, status, login, logout]
  );

  return <AuthContext.Provider value={memoizedValue}>{children}</AuthContext.Provider>;
}
