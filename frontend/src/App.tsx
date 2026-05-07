import { useState } from 'react';
import { RecitationPage } from './pages/RecitationPage';
import QuranReaderPage from './pages/QuranReaderPage';

type Mode = 'single' | 'quran';

function App() {
  const [mode, setMode] = useState<Mode>('single');

  if (mode === 'quran') {
    return <QuranReaderPage onBack={() => setMode('single')} />;
  }

  return <RecitationPage onSwitchMode={() => setMode('quran')} />;
}

export default App;
