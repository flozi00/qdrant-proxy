import { AppProvider, useApp } from './store';
import Layout from './components/Layout';
import LoginScreen from './components/LoginScreen';

function AppContent() {
  const { isLoggedIn } = useApp();
  return isLoggedIn ? <Layout /> : <LoginScreen />;
}

export default function App() {
  return (
    <AppProvider>
      <AppContent />
    </AppProvider>
  );
}
