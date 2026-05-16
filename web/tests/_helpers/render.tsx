// web/tests/_helpers/render.tsx
import type { ReactElement } from 'react';
import { render as rtlRender, type RenderOptions } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { SelectedRefProvider } from '@/state/selectedRef';

export function render(ui: ReactElement, options?: RenderOptions) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });
  return rtlRender(
    <QueryClientProvider client={queryClient}>
      <SelectedRefProvider>{ui}</SelectedRefProvider>
    </QueryClientProvider>,
    options,
  );
}

export * from '@testing-library/react';
