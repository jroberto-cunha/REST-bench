"""CLI for the restaurant management simulation.

Designed for single-shot invocations: each command reads the persistent
game state, performs one action or one day of simulation, prints a
compact JSON/text response, and exits. Intended to be driven by an
agent loop.

Usage:
    python -m rest_sim <command> [args...]
Run `python -m rest_sim help` for a list of commands.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np

from . import game_state as gs
from . import observability as obs
from . import dashboard_client
from .day_sim import run_one_day, pregenerate_reservations


def _print(data, as_json: bool = True):
    if isinstance(data, str):
        print(data)
        return
    print(json.dumps(data, indent=2, default=str))


def _push_decision(state: dict, kind: str, payload: dict) -> None:
    """Mirror the most recent decision to the live dashboard (best-effort)."""
    dashboard_client.post("decision", {
        "kind": kind,
        "payload": payload,
        "day_index": state.get("day_index"),
        "date": gs.current_date(state) if not gs.is_done(state) else None,
    })


def cmd_init(args):
    dashboard_client.ensure_running(open_browser=not args.no_dashboard_browser)
    # Wipe any prior decisions log so the JSONL is per-run.
    log_path = gs._decisions_log_path()
    if os.path.exists(log_path):
        os.remove(log_path)
    state = gs.default_state(
        seed=args.seed, start_date=args.start_date,
        total_days=args.days, starting_cash=args.cash,
    )
    rng = np.random.default_rng(args.seed)
    pregenerate_reservations(state, rng)
    # Persist the RNG state so repeated runs are reproducible
    state["_rng_seed_counter"] = args.seed + 1
    gs.save(state)
    dashboard_client.post("init", {
        "seed": state["seed"],
        "start_date": state["start_date"],
        "total_days": state["total_days"],
        "starting_cash_eur": state["starting_cash_eur"],
        "reservations_pregenerated": len(state["reservations_all"]),
        "status": obs.status_snapshot(state),
        "inventory": dict(state["inventory"]),
        "table_layout": dict(state["table_layout"]),
    })
    _print({
        "ok": True, "message": "Game initialized.",
        "start_date": state["start_date"], "total_days": state["total_days"],
        "starting_cash_eur": state["starting_cash_eur"],
        "reservations_pregenerated": len(state["reservations_all"]),
    })


def cmd_status(args):
    state = gs.load()
    _print(obs.status_snapshot(state, full=getattr(args, "full", False)))


def cmd_kpis(args):
    state = gs.load()
    rows = state["daily_kpis_history"]
    if args.days:
        rows = rows[-args.days:]
    # Strip hour_heat for brevity
    out = [{k: v for k, v in r.items() if k != "hour_heat"} for r in rows]
    _print(out)


def cmd_pnl(args):
    state = gs.load()
    _print(obs.pnl_summary(state, args.days))


def cmd_attribution(args):
    _print(obs.attribution(gs.load()))


def cmd_heatmap(args):
    state = gs.load()
    h = obs.heatmap(state, args.days)
    _print(h["ascii"])


def cmd_cohort(args):
    _print(obs.cohort_breakdown(gs.load(), args.days))


def cmd_menu(args):
    state = gs.load()
    menu = gs.effective_menu(state)
    if args.low_margin:
        menu = sorted(menu, key=lambda m: m["margin_pct"])[:args.low_margin]
    _print([{k: m[k] for k in ("id", "name", "category", "price_eur",
                                "food_cost_eur", "margin_pct",
                                "popularity_weight")} for m in menu])


def cmd_set_price(args):
    state = gs.load()
    state["price_overrides"][args.item_id] = float(args.price)
    gs.log_decision(state, "set_price", {"item_id": args.item_id, "price": args.price})
    gs.save(state)
    _push_decision(state, "set_price", {"item_id": args.item_id, "price": args.price})
    _print({"ok": True, "item_id": args.item_id, "new_price_eur": args.price})


def cmd_remove_item(args):
    state = gs.load()
    if args.item_id not in state["menu_removed"]:
        state["menu_removed"].append(args.item_id)
    gs.log_decision(state, "remove_item", {"item_id": args.item_id})
    gs.save(state)
    _push_decision(state, "remove_item", {"item_id": args.item_id})
    _print({"ok": True, "removed": args.item_id})


def cmd_add_item(args):
    state = gs.load()
    new_id = f"X{len(state['menu_added'])+1:03d}"
    state["menu_added"].append({
        "id": new_id, "name": args.name, "category": args.category,
        "price": float(args.price), "cost": float(args.cost),
    })
    state["inventory"][new_id] = gs.DEFAULT_STOCK
    gs.log_decision(state, "add_item", {"id": new_id, "name": args.name})
    gs.save(state)
    _push_decision(state, "add_item", {"id": new_id, "name": args.name,
                                        "category": args.category, "price": args.price})
    _print({"ok": True, "item_id": new_id})


def cmd_inventory(args):
    state = gs.load()
    inv = state["inventory"]
    rows = sorted(inv.items(), key=lambda kv: kv[1])
    if args.low:
        rows = [(k, v) for k, v in rows if v < args.low]
    _print({"inventory": dict(rows),
            "ingredient_price_multiplier": state["ingredient_price_multiplier"]})


def cmd_restock(args):
    state = gs.load()
    from .economy import restock_cost
    cost = restock_cost(state, args.item_id, args.qty)
    if cost > state["cash_eur"]:
        _print({"ok": False, "error": "insufficient cash",
                "required_eur": cost, "cash_eur": state["cash_eur"]})
        return
    state["inventory"][args.item_id] = state["inventory"].get(args.item_id, 0) + int(args.qty)
    state["cash_eur"] = round(state["cash_eur"] - cost, 2)
    gs.log_decision(state, "restock",
                    {"item_id": args.item_id, "qty": args.qty, "cost_eur": cost})
    gs.save(state)
    _push_decision(state, "restock",
                   {"item_id": args.item_id, "qty": args.qty, "cost_eur": cost})
    _print({"ok": True, "item_id": args.item_id, "qty_added": args.qty,
            "cost_eur": cost, "cash_eur": state["cash_eur"]})


def cmd_staffing(args):
    state = gs.load()
    from .economy import scheduled_staff_for_date, payroll_for_date
    date = args.date or gs.current_date(state)
    _print({
        "date": date,
        "effective_by_role": scheduled_staff_for_date(state, date),
        "override": state["staffing_override"].get(date, {}),
        "estimated_payroll_eur": payroll_for_date(state, date),
    })


def cmd_set_staff(args):
    state = gs.load()
    override = state["staffing_override"].setdefault(args.date, {})
    override[args.role] = int(args.count)
    gs.log_decision(state, "set_staff",
                    {"date": args.date, "role": args.role, "count": args.count})
    gs.save(state)
    _push_decision(state, "set_staff",
                   {"date": args.date, "role": args.role, "count": args.count})
    _print({"ok": True, "date": args.date, "role": args.role, "count": args.count})


def cmd_reservations(args):
    state = gs.load()
    by_date: dict[str, int] = {}
    for r in state["reservations_all"]:
        by_date[r["date"]] = by_date.get(r["date"], 0) + 1
    _print({"per_date": by_date, "caps": state["reservation_caps"]})


def cmd_set_cap(args):
    state = gs.load()
    state["reservation_caps"][args.date] = int(args.cap)
    gs.log_decision(state, "set_reservation_cap",
                    {"date": args.date, "cap": args.cap})
    gs.save(state)
    _push_decision(state, "set_reservation_cap",
                   {"date": args.date, "cap": args.cap})
    _print({"ok": True, "date": args.date, "cap": args.cap})


def cmd_happy_hour(args):
    state = gs.load()
    start_date = args.from_date or gs.current_date(state)
    end = (datetime.fromisoformat(start_date)
           + timedelta(days=args.days - 1)).date().isoformat()
    cats = [c.strip() for c in args.categories.split(",") if c.strip()] if args.categories else []
    hh = {
        "start_hour": float(args.start),
        "end_hour": float(args.end),
        "discount_pct": float(args.discount),
        "categories": cats,
        "from_date": start_date,
        "until_date": end,
    }
    state["happy_hours"].append(hh)
    gs.log_decision(state, "happy_hour", hh)
    gs.save(state)
    _push_decision(state, "happy_hour", hh)
    _print({"ok": True, **hh})


def cmd_promo(args):
    state = gs.load()
    start_date = gs.current_date(state)
    end = (datetime.fromisoformat(start_date)
           + timedelta(days=args.days - 1)).date().isoformat()
    state["active_promos"].append({
        "category": args.category,
        "discount_pct": float(args.discount),
        "until": end,
    })
    gs.log_decision(state, "promo",
                    {"category": args.category, "discount_pct": args.discount,
                     "until": end})
    gs.save(state)
    _push_decision(state, "promo",
                   {"category": args.category, "discount_pct": args.discount,
                    "until": end})
    _print({"ok": True, "category": args.category, "discount_pct": args.discount,
            "until": end})


def cmd_marketing(args):
    state = gs.load()
    date = args.date or gs.current_date(state)
    state["marketing_spend_by_day"][date] = (
        state["marketing_spend_by_day"].get(date, 0.0) + float(args.amount)
    )
    gs.log_decision(state, "marketing",
                    {"date": date, "amount_eur": args.amount})
    gs.save(state)
    _push_decision(state, "marketing",
                   {"date": date, "amount_eur": args.amount})
    _print({"ok": True, "date": date,
            "total_spend_eur": state["marketing_spend_by_day"][date]})


def cmd_loyalty(args):
    state = gs.load()
    state["loyalty_enabled"] = (args.action == "on")
    gs.log_decision(state, "loyalty", {"enabled": state["loyalty_enabled"]})
    gs.save(state)
    _push_decision(state, "loyalty", {"enabled": state["loyalty_enabled"]})
    _print({"ok": True, "loyalty_enabled": state["loyalty_enabled"]})


def cmd_tables(args):
    state = gs.load()
    _print({"layout": state["table_layout"]})


def cmd_convert_table(args):
    state = gs.load()
    tid = args.table_id
    if tid not in state["table_layout"]:
        _print({"ok": False, "error": f"unknown table {tid}"})
        return
    from .tuning import TABLE
    conversion_cost = TABLE.CONVERSION_COST_EUR
    if state["cash_eur"] < conversion_cost:
        _print({"ok": False, "error": "insufficient cash"})
        return
    old = state["table_layout"][tid]
    state["table_layout"][tid] = int(args.seats)
    state["cash_eur"] = round(state["cash_eur"] - conversion_cost, 2)
    state["table_layout_changes"].append({
        "date": gs.current_date(state) if not gs.is_done(state) else None,
        "table_id": tid, "from_seats": old, "to_seats": int(args.seats),
    })
    gs.log_decision(state, "convert_table",
                    {"table_id": tid, "from": old, "to": args.seats})
    gs.save(state)
    _push_decision(state, "convert_table",
                   {"table_id": tid, "from": old, "to": args.seats,
                    "cost_eur": conversion_cost})
    _print({"ok": True, "table_id": tid, "seats": args.seats,
            "cost_eur": conversion_cost})


def cmd_competitor(args):
    state = gs.load()
    state["competitor_active"] = (args.action == "on")
    if state["competitor_active"]:
        state["competitor_active_from"] = gs.current_date(state)
    gs.save(state)
    _print({"ok": True, "competitor_active": state["competitor_active"]})


def cmd_advance(args):
    state = gs.load()
    if state.get("bankrupt", False):
        _print({"done": True, "bankrupt": True,
                "bankrupt_on_date": state.get("bankrupt_on_date"),
                "message": "Restaurant is bankrupt; simulation halted.",
                "scorecard": obs.scorecard(state)})
        return
    if gs.is_done(state):
        _print({"done": True, "message": "Simulation complete.",
                "scorecard": obs.scorecard(state)})
        return
    # Derive a day-specific rng seed so each day is deterministic
    day_seed = state["seed"] + state["day_index"] * 1_000_003
    rng = np.random.default_rng(day_seed)
    result = run_one_day(state, rng)
    gs.log_day_advance(state, result["kpi"])
    gs.save(state)
    payload = {
        "ok": True, "day_advanced": result["kpi"]["date"],
        "kpi": {k: v for k, v in result["kpi"].items() if k != "hour_heat"},
        "pnl": result["pnl"],
        "weather": result["weather"],
        "incident": result["incident"],
        "outages": result["outages"],
        "status": obs.status_snapshot(state),
        "pnl_trailing_7d": obs.pnl_summary(state, 7),
        "cohort_7d": obs.cohort_breakdown(state, 7),
        "attribution": obs.attribution(state),
        "heatmap_ascii": obs.heatmap(state, 14)["ascii"],
        "recent_decisions": state["decisions_log"][-5:],
        "done": gs.is_done(state),
    }
    if state.get("bankrupt", False):
        payload["bankrupt"] = True
        payload["bankrupt_on_date"] = state.get("bankrupt_on_date")
    if gs.is_done(state) or state.get("bankrupt", False):
        payload["scorecard"] = obs.scorecard(state)
    # Dashboard-only enrichment: full inventory snapshot for live display.
    dash_payload = dict(payload)
    dash_payload["inventory"] = dict(state["inventory"])
    dashboard_client.post("advance", dash_payload)
    _print(payload)


def cmd_done(args):
    state = gs.load()
    _print({"done": gs.is_done(state),
            "day_index": state["day_index"],
            "total_days": state["total_days"]})


def cmd_scorecard(args):
    state = gs.load()
    _print(obs.scorecard(state))


def cmd_decisions(args):
    state = gs.load()
    rows = state["decisions_log"]
    if args.last:
        rows = rows[-args.last:]
    _print(rows)


def cmd_cohorts(args):
    """Phase 2.1 — banded cohort sizes (counts hidden; only L/N/H bands shown)."""
    state = gs.load()
    from . import cohorts as cohort_mod
    bands = cohort_mod.cohort_bands(state.get("cohorts", {}))
    _print({"cohort_sizes_band": bands})


def cmd_news(args):
    """Phase 2.4 — pending supplier alerts."""
    state = gs.load()
    _print({
        "supplier_news": [
            {
                "fire_on_date": n["fire_on_date"],
                "lead_days": n["lead_days"],
                "headline": n["headline"],
                "predicted_magnitude_range": n["predicted_magnitude_range"],
                "confidence": n["confidence"],
            }
            for n in state.get("supplier_news", [])
        ],
    })


def cmd_reservations_next(args):
    """Phase 0.4 — upcoming reservation counts within a window (counts only)."""
    state = gs.load()
    from datetime import datetime, timedelta
    today = gs.current_date(state) if not gs.is_done(state) else gs.date_for(state, state["total_days"]-1)
    horizon = (datetime.fromisoformat(today)
               + timedelta(days=args.days - 1)).date().isoformat()
    by_date: dict[str, int] = {}
    for r in state.get("reservations_all", []):
        if today <= r["date"] <= horizon:
            by_date[r["date"]] = by_date.get(r["date"], 0) + 1
    _print({
        "window_from": today, "window_to": horizon,
        "per_date_counts": by_date,
        "caps": {d: c for d, c in state["reservation_caps"].items()
                 if today <= d <= horizon},
    })


def cmd_dashboard(args):
    from . import dashboard as _dash
    _dash.serve(port=args.port, open_browser=not args.no_browser)


def cmd_help(args):
    lines = [
        "Commands (all persist state to game/state.json):",
        "  init [--seed N] [--start-date YYYY-MM-DD] [--days 30] [--cash 20000]",
        "  status                       — manager view (last 7d KPIs, cohort bands, news)",
        "  kpis [--days N]              — daily KPI history",
        "  pnl [--days N]               — aggregated P&L",
        "  attribution                  — last-7 vs prior-7 deltas",
        "  heatmap [--days N]           — arrivals by dow × hour",
        "  cohort [--days 7]            — reservation vs walk-in share (visit-level)",
        "  cohorts                      — banded customer-population view (Phase 2.1)",
        "  news                         — supplier alerts ahead of price shocks (Phase 2.4)",
        "  menu [--low-margin N]        — menu with effective prices / margins",
        "  set-price ITEM_ID PRICE",
        "  add-item NAME CATEGORY PRICE COST",
        "  remove-item ITEM_ID",
        "  inventory [--low N]",
        "  restock ITEM_ID QTY",
        "  staffing [--date D]",
        "  set-staff DATE ROLE COUNT",
        "  reservations                 — counts per date (full horizon)",
        "  reservations-next [--days 7] — counts within near-term window",
        "  set-cap DATE CAP",
        "  promo CATEGORY DISCOUNT_PCT --days N",
        "  happy-hour --start H --end H --discount D [--categories cats] [--days N]  (Phase 3.1)",
        "  marketing AMOUNT [--date D]",
        "  loyalty {on|off}",
        "  tables",
        "  convert-table TABLE_ID SEATS",
        "  competitor {on|off}",
        "  advance                      — simulate next day",
        "  done                         — is the run over?",
        "  scorecard                    — final summary (incl. score breakdown)",
        "  decisions [--last N]",
        "  dashboard [--port 8765] [--no-browser]",
    ]
    print("\n".join(lines))


def build_parser():
    p = argparse.ArgumentParser(prog="rest_sim")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init");      s.set_defaults(fn=cmd_init)
    s.add_argument("--seed", type=int, default=20260423)
    s.add_argument("--start-date", default="2026-04-20")
    s.add_argument("--days", type=int, default=30)
    s.add_argument("--cash", type=float, default=20000.0)
    s.add_argument("--no-dashboard-browser", action="store_true",
                   help="Spawn the dashboard but don't auto-open the browser")

    s = sub.add_parser("status"); s.set_defaults(fn=cmd_status)
    s.add_argument("--full", action="store_true",
                   help="Return the unmasked engine state (tests/dashboard only — "
                        "agents must NOT use this).")

    s = sub.add_parser("kpis");      s.set_defaults(fn=cmd_kpis)
    s.add_argument("--days", type=int, default=7)

    s = sub.add_parser("pnl");       s.set_defaults(fn=cmd_pnl)
    s.add_argument("--days", type=int, default=None)

    sub.add_parser("attribution").set_defaults(fn=cmd_attribution)

    s = sub.add_parser("heatmap");   s.set_defaults(fn=cmd_heatmap)
    s.add_argument("--days", type=int, default=None)

    s = sub.add_parser("cohort");    s.set_defaults(fn=cmd_cohort)
    s.add_argument("--days", type=int, default=7)

    s = sub.add_parser("menu");      s.set_defaults(fn=cmd_menu)
    s.add_argument("--low-margin", type=int, default=None)

    s = sub.add_parser("set-price"); s.set_defaults(fn=cmd_set_price)
    s.add_argument("item_id"); s.add_argument("price", type=float)

    s = sub.add_parser("add-item");  s.set_defaults(fn=cmd_add_item)
    s.add_argument("name"); s.add_argument("category")
    s.add_argument("price", type=float); s.add_argument("cost", type=float)

    s = sub.add_parser("remove-item"); s.set_defaults(fn=cmd_remove_item)
    s.add_argument("item_id")

    s = sub.add_parser("inventory"); s.set_defaults(fn=cmd_inventory)
    s.add_argument("--low", type=int, default=None)

    s = sub.add_parser("restock");   s.set_defaults(fn=cmd_restock)
    s.add_argument("item_id"); s.add_argument("qty", type=int)

    s = sub.add_parser("staffing");  s.set_defaults(fn=cmd_staffing)
    s.add_argument("--date", default=None)

    s = sub.add_parser("set-staff"); s.set_defaults(fn=cmd_set_staff)
    s.add_argument("date"); s.add_argument("role"); s.add_argument("count", type=int)

    sub.add_parser("reservations").set_defaults(fn=cmd_reservations)

    s = sub.add_parser("reservations-next"); s.set_defaults(fn=cmd_reservations_next)
    s.add_argument("--days", type=int, default=7,
                   help="Number of days from today to summarize")

    sub.add_parser("cohorts").set_defaults(fn=cmd_cohorts)
    sub.add_parser("news").set_defaults(fn=cmd_news)

    s = sub.add_parser("set-cap");   s.set_defaults(fn=cmd_set_cap)
    s.add_argument("date"); s.add_argument("cap", type=int)

    s = sub.add_parser("promo");     s.set_defaults(fn=cmd_promo)
    s.add_argument("category"); s.add_argument("discount", type=float)
    s.add_argument("--days", type=int, default=3)

    s = sub.add_parser("happy-hour"); s.set_defaults(fn=cmd_happy_hour)
    s.add_argument("--start", type=float, required=True,
                   help="Start hour (e.g. 16 = 4pm)")
    s.add_argument("--end", type=float, required=True,
                   help="End hour (e.g. 18 = 6pm)")
    s.add_argument("--discount", type=float, required=True,
                   help="Discount fraction (0.20 = 20%%)")
    s.add_argument("--categories", default=None,
                   help="Comma-separated categories; default = all")
    s.add_argument("--days", type=int, default=7)
    s.add_argument("--from-date", default=None,
                   dest="from_date", help="ISO date; default = today")

    s = sub.add_parser("marketing"); s.set_defaults(fn=cmd_marketing)
    s.add_argument("amount", type=float); s.add_argument("--date", default=None)

    s = sub.add_parser("loyalty");   s.set_defaults(fn=cmd_loyalty)
    s.add_argument("action", choices=["on", "off"])

    sub.add_parser("tables").set_defaults(fn=cmd_tables)

    s = sub.add_parser("convert-table"); s.set_defaults(fn=cmd_convert_table)
    s.add_argument("table_id"); s.add_argument("seats", type=int)

    s = sub.add_parser("competitor"); s.set_defaults(fn=cmd_competitor)
    s.add_argument("action", choices=["on", "off"])

    sub.add_parser("advance").set_defaults(fn=cmd_advance)
    sub.add_parser("done").set_defaults(fn=cmd_done)
    sub.add_parser("scorecard").set_defaults(fn=cmd_scorecard)

    s = sub.add_parser("decisions"); s.set_defaults(fn=cmd_decisions)
    s.add_argument("--last", type=int, default=None)

    s = sub.add_parser("dashboard"); s.set_defaults(fn=cmd_dashboard)
    s.add_argument("--port", type=int,
                   default=int(os.environ.get("REST_DASHBOARD_PORT", "8765")))
    s.add_argument("--no-browser", action="store_true")

    sub.add_parser("help").set_defaults(fn=cmd_help)

    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
