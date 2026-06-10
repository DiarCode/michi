import { describe, it, expect, beforeEach, vi } from "vitest";

const classListMock = {
  add: vi.fn(),
  remove: vi.fn(),
  contains: vi.fn(),
};

// Mock document.documentElement before import
Object.defineProperty(document, "documentElement", {
  value: { classList: classListMock },
  writable: true,
});

// Mock localStorage for zustand persist
const store: Record<string, string> = {};
vi.stubGlobal("localStorage", {
  getItem: (key: string) => store[key] ?? null,
  setItem: (key: string, value: string) => { store[key] = value; },
  removeItem: (key: string) => { delete store[key]; },
  clear: () => { Object.keys(store).forEach((k) => delete store[k]); },
});

import { useThemeStore } from "./themeStore";

describe("themeStore", () => {
  beforeEach(() => {
    classListMock.add.mockClear();
    classListMock.remove.mockClear();
    useThemeStore.setState({ theme: "light", resolvedTheme: "light" });
  });

  it("defaults to light theme", () => {
    const { theme, resolvedTheme } = useThemeStore.getState();
    expect(theme).toBe("light");
    expect(resolvedTheme).toBe("light");
  });

  it("sets dark theme and applies dark class to document", () => {
    useThemeStore.getState().setTheme("dark");
    expect(useThemeStore.getState().theme).toBe("dark");
    expect(useThemeStore.getState().resolvedTheme).toBe("dark");
    expect(classListMock.add).toHaveBeenCalledWith("dark");
    expect(classListMock.remove).toHaveBeenCalledWith("light");
  });

  it("sets light theme and applies light class to document", () => {
    useThemeStore.getState().setTheme("dark");
    classListMock.add.mockClear();
    classListMock.remove.mockClear();

    useThemeStore.getState().setTheme("light");
    expect(useThemeStore.getState().theme).toBe("light");
    expect(useThemeStore.getState().resolvedTheme).toBe("light");
    expect(classListMock.add).toHaveBeenCalledWith("light");
    expect(classListMock.remove).toHaveBeenCalledWith("dark");
  });

  it("resolves system theme based on matchMedia (light default)", () => {
    useThemeStore.getState().setTheme("system");
    expect(useThemeStore.getState().resolvedTheme).toBe("light");
  });

  it("toggles from dark back to light correctly", () => {
    useThemeStore.getState().setTheme("dark");
    expect(useThemeStore.getState().resolvedTheme).toBe("dark");

    useThemeStore.getState().setTheme("light");
    expect(useThemeStore.getState().resolvedTheme).toBe("light");
  });
});