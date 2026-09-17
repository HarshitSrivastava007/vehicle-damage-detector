import { createContext, type ReactNode } from "react"
import { useQuery } from "@tanstack/react-query"
import { ApiError, api } from "../api/client"
import type { SessionUser } from "../api/types"

export interface AuthContextValue {
  user: SessionUser | null
  isLoading: boolean
}

export const AuthContext = createContext<AuthContextValue | undefined>(undefined)

async function fetchSession(): Promise<SessionUser | null> {
  try {
    return await api.get<SessionUser>("/auth/session")
  } catch (err) {
    if (err instanceof ApiError && err.status === 401) {
      return null
    }
    throw err
  }
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const { data, isLoading } = useQuery({
    queryKey: ["session"],
    queryFn: fetchSession,
    retry: false,
    staleTime: Infinity,
  })

  return <AuthContext.Provider value={{ user: data ?? null, isLoading }}>{children}</AuthContext.Provider>
}
