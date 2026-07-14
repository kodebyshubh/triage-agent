# TriageForge Frontend

Minimal React + Vite + Tailwind UI for TriageForge. One page: submit a ticket,
see the raw response, the agent trace timeline, and the final result card.

Run the backend first (see the root [README](../README.md)), then:

```
npm install
npm run dev
```

Opens on `http://localhost:5173`. Expects the backend at `http://127.0.0.1:8000`
(see `API_URL` in `src/App.jsx`).
