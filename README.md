# A-share sector data collector

This repository produces a public `data/latest.json` evidence pack for a personal A-share industry-board research workflow.

- Classification: Eastmoney industry boards, not Shenwan or CITIC industries.
- Data source: AkShare public-data adapters.
- Schedule: 11:38 Asia/Shanghai on weekdays, subject to GitHub Actions scheduling delay.
- Scope: industry boards only. It does not analyse or recommend individual stocks.
- Safety: a missing, stale, or incomplete midday snapshot sets `data_status` to `unavailable`; consumers must not generate direction conclusions in that case.

The generated evidence distinguishes a noon short-term snapshot from completed daily bars used by medium- and long-term indicators. It contains no API keys, personal data, or trading instructions.
