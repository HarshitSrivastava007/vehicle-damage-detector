import { createBrowserRouter } from "react-router-dom"
import { RequireAdmin } from "./auth/RequireAdmin"
import { RequireAuth } from "./auth/RequireAuth"
import { AppShell } from "./components/AppShell"
import { AdminUsersPage } from "./pages/AdminUsersPage"
import { ApiKeysPage } from "./pages/ApiKeysPage"
import { DashboardPage } from "./pages/DashboardPage"
import { DetectorTestPage } from "./pages/DetectorTestPage"
import { LoginPage } from "./pages/LoginPage"
import { NotFoundPage } from "./pages/NotFoundPage"

export const router = createBrowserRouter([
  { path: "/login", element: <LoginPage /> },
  {
    element: <RequireAuth />,
    children: [
      {
        element: <AppShell />,
        children: [
          { path: "/", element: <DashboardPage /> },
          { path: "/keys", element: <ApiKeysPage /> },
          { path: "/detect", element: <DetectorTestPage /> },
          {
            element: <RequireAdmin />,
            children: [{ path: "/admin", element: <AdminUsersPage /> }],
          },
        ],
      },
    ],
  },
  { path: "*", element: <NotFoundPage /> },
])
