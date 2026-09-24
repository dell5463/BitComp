---
name: powershell-utf8-edits
description: Windows PowerShell 5.1 Get-Content/Set-Content mangles UTF-8 source files; use the Edit/Write tools for content edits
metadata:
  type: feedback
---

Do not edit source/docs via `Get-Content -Raw ... .Replace(...) | Set-Content` in Windows PowerShell 5.1.

**Why:** `Get-Content` without `-Encoding UTF8` reads UTF-8 as ANSI, so non-ASCII characters (∞, ↑, −) were silently corrupted into mojibake (`âˆž`) in BitLaya plot labels during a 2026-09-24 session; `Set-Content -Encoding utf8` also adds a BOM.

**How to apply:** use the Edit/Write tools for any file content change. If a scripted multi-replace is unavoidable, pass `-Encoding UTF8` to Get-Content and afterwards grep for `â|Ã|Â` to confirm nothing was mangled.
