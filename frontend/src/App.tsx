import { useState } from 'react';
import { RecitationPage } from './pages/RecitationPage';
import FollowAlongPage from './pages/FollowAlongPage';

type Mode = 'single' | 'follow';

function App() {
  const [mode, setMode] = useState<Mode>('single');

  if (mode === 'follow') {
    return <FollowAlongPage onBack={() => setMode('single')} />;
  }

  return <RecitationPage onSwitchMode={() => setMode('follow')} />;
}

export default App;
