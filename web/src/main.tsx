import './styles/global.css';
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { SelectedRefProvider } from '@/state/selectedRef';
import { IconSprite } from '@/icons/sprite';
import { App } from './App';

const queryClient = new QueryClient({
  defaultOptions: { queries: { refetchOnWindowFocus: false } },
});

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <SelectedRefProvider>
        <IconSprite />
        <App />
      </SelectedRefProvider>
    </QueryClientProvider>
  </StrictMode>,
);
