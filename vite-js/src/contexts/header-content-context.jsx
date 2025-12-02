import { useMemo, useState, useContext, createContext } from 'react';

// ----------------------------------------------------------------------

const HeaderContentContext = createContext(null);

// ----------------------------------------------------------------------

export function HeaderContentProvider({ children }) {
  const [headerContent, setHeaderContent] = useState(null);
  const [hideRightButtons, setHideRightButtons] = useState(false);

  const value = useMemo(() => ({
    headerContent,
    setHeaderContent,
    hideRightButtons,
    setHideRightButtons,
  }), [headerContent, hideRightButtons]);

  return (
    <HeaderContentContext.Provider value={value}>
      {children}
    </HeaderContentContext.Provider>
  );
}

export function useHeaderContent() {
  const context = useContext(HeaderContentContext);
  if (!context) {
    // Return default values if context is not available
    return { 
      headerContent: null, 
      setHeaderContent: () => {},
      hideRightButtons: false,
      setHideRightButtons: () => {},
    };
  }
  return context;
}
