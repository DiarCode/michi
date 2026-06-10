import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { ConnectionIndicator } from "../components/ConnectionIndicator";
import { useConnectionStore } from "../stores/connectionStore";

// Mock the store — useConnectionStore is called with a selector (s) => s.wsState
// so we need to return a function that calls the selector on our mock state.
vi.mock("../stores/connectionStore", () => {
  let mockState: Record<string, unknown> = { wsState: "disconnected", connected: false };
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const mockStore: any = vi.fn((selector: (s: Record<string, unknown>) => unknown) => selector(mockState));
  mockStore.getState = () => mockState;
  // Allow tests to update mock state
  (mockStore as unknown as { __setMockState: (s: Record<string, unknown>) => void }).__setMockState = (s: Record<string, unknown>) => {
    mockState = s;
  };
  return { useConnectionStore: mockStore };
});

const setMockState = (useConnectionStore as unknown as { __setMockState: (s: Record<string, unknown>) => void }).__setMockState;

describe("ConnectionIndicator", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders green dot when connected", () => {
    setMockState({ wsState: "connected", connected: true });
    render(<ConnectionIndicator />);
    expect(screen.getByTitle("WebSocket: connected")).toBeInTheDocument();
    expect(screen.getByText("Live")).toBeInTheDocument();
  });

  it("renders yellow pulsing dot when connecting", () => {
    setMockState({ wsState: "connecting", connected: false });
    render(<ConnectionIndicator />);
    expect(screen.getByTitle("WebSocket: connecting")).toBeInTheDocument();
    expect(screen.getByText("Connecting")).toBeInTheDocument();
  });

  it("renders red pulsing dot when disconnected", () => {
    setMockState({ wsState: "disconnected", connected: false });
    render(<ConnectionIndicator />);
    expect(screen.getByTitle("WebSocket: disconnected")).toBeInTheDocument();
    expect(screen.getByText("Offline")).toBeInTheDocument();
  });
});