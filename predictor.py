import json
import pandas as pd
import numpy as np

N_ITERAZIONI     = 10_000
SIGMA_BASE       = 0.10
SIGMA_METEO      = 0.14   # rumore extra in caso di meteo avverso
NAZIONI_OSPITANTI = {'United States', 'Canada', 'Mexico'}


def carica_squadre():
    return pd.read_csv('data/teams.csv')


def calcola_forza(squadra):
    # Rank invertito: rank 1 = punteggio 1.0, rank 200 = 0.0
    rank_score  = 1 - (squadra['fifa_rank'] - 1) / 199
    gol_fatti   = min(squadra['goals_scored_avg'] / 3.0, 1.0)
    # Gol subiti invertito: meno gol subiti = punteggio più alto
    gol_subiti  = max(1 - squadra['goals_conceded_avg'] / 2.0, 0.0)
    win_rate    = squadra['win_rate_12m']
    knockout    = squadra['knockout_win_rate']

    return (0.25 * rank_score +
            0.25 * gol_fatti +
            0.25 * gol_subiti +
            0.15 * win_rate +
            0.10 * knockout)


def applica_dinamiche(forza_a, forza_b, squadra_a, squadra_b):
    fa, fb = forza_a, forza_b

    # Infortunio: 5% per squadra → forza -10%
    if np.random.random() < 0.05:
        fa *= 0.90
    if np.random.random() < 0.05:
        fb *= 0.90

    # Espulsione: 8% per squadra → forza -15%
    if np.random.random() < 0.08:
        fa *= 0.85
    if np.random.random() < 0.08:
        fb *= 0.85

    # Meteo avverso: 10% per partita → forza -5% per entrambe, più casualità
    sigma = SIGMA_BASE
    if np.random.random() < 0.10:
        fa   *= 0.95
        fb   *= 0.95
        sigma = SIGMA_METEO

    # Vantaggio tifosi di casa (solo USA, Canada, Messico in ogni partita del torneo)
    if squadra_a['team'] in NAZIONI_OSPITANTI:
        fa *= 1.08
    if squadra_b['team'] in NAZIONI_OSPITANTI:
        fb *= 1.08

    # Giornata di grazia: 3% per squadra → forza +12%
    if np.random.random() < 0.03:
        fa *= 1.12
    if np.random.random() < 0.03:
        fb *= 1.12

    return fa, fb, sigma


def simula_partita(squadra_a, squadra_b):
    forza_a = calcola_forza(squadra_a)
    forza_b = calcola_forza(squadra_b)

    forza_a, forza_b, sigma = applica_dinamiche(forza_a, forza_b, squadra_a, squadra_b)

    rumore_a = np.random.normal(0, sigma, N_ITERAZIONI)
    rumore_b = np.random.normal(0, sigma, N_ITERAZIONI)

    return np.sum((forza_a + rumore_a) > (forza_b + rumore_b)) / N_ITERAZIONI


def simula_gol(squadra):
    return int(np.random.poisson(squadra['goals_scored_avg']))


def simula_torneo(squadre):
    # --- Fase a gironi (72 partite) ---
    risultati = {}

    for nome, gruppo in squadre.groupby('girone'):
        righe = [row for _, row in gruppo.iterrows()]
        punti = {r['team']: 0 for r in righe}

        for i in range(4):
            for j in range(i + 1, 4):
                a, b = righe[i], righe[j]
                prob_a = simula_partita(a, b)
                delta = abs(calcola_forza(a) - calcola_forza(b))
                prob_pareggio = max(0.10, 0.28 - 0.20 * delta)

                r = np.random.random()
                if r < prob_a * (1 - prob_pareggio):
                    punti[a['team']] += 3
                elif r < prob_a * (1 - prob_pareggio) + prob_pareggio:
                    punti[a['team']] += 1
                    punti[b['team']] += 1
                else:
                    punti[b['team']] += 3

        classifica = sorted(righe,
                            key=lambda s: (punti[s['team']], calcola_forza(s)),
                            reverse=True)
        risultati[nome] = {
            'primo':       classifica[0],
            'secondo':     classifica[1],
            'terzo':       classifica[2],
            'punti_terzo': punti[classifica[2]['team']],
        }

    # Pool delle 8 migliori terze: ogni slot del tabellone pesca da qui
    terze_ordinate = sorted(
        [(n, r['terzo'], r['punti_terzo']) for n, r in risultati.items()],
        key=lambda x: (x[2], calcola_forza(x[1])),
        reverse=True
    )[:8]
    terze_pool = {n: (sq, pt) for n, sq, pt in terze_ordinate}

    def miglior_terza(gironi_ammessi):
        # Pesca la terza con più punti tra i gironi ammessi e la rimuove dal pool
        candidati = [(n, sq, pt) for n, (sq, pt) in terze_pool.items()
                     if n in gironi_ammessi]
        if not candidati:  # fallback: nessun girone ammesso disponibile
            candidati = [(n, sq, pt) for n, (sq, pt) in terze_pool.items()]
        scelta = max(candidati, key=lambda x: (x[2], calcola_forza(x[1])))
        del terze_pool[scelta[0]]
        return scelta[1]

    def p(a, b):
        return a if np.random.random() < simula_partita(a, b) else b

    g = risultati

    # --- Sedicesimi (match 73–88, 16 partite) ---
    m73 = p(g['A']['secondo'], g['B']['secondo'])
    m74 = p(g['E']['primo'],   miglior_terza({'A', 'B', 'C', 'D', 'F'}))
    m75 = p(g['F']['primo'],   g['C']['secondo'])
    m76 = p(g['C']['primo'],   g['F']['secondo'])
    m77 = p(g['I']['primo'],   miglior_terza({'C', 'D', 'F', 'G', 'H'}))
    m78 = p(g['E']['secondo'], g['I']['secondo'])
    m79 = p(g['A']['primo'],   miglior_terza({'C', 'E', 'F', 'H', 'I'}))
    m80 = p(g['L']['primo'],   miglior_terza({'E', 'H', 'I', 'J', 'K'}))
    m81 = p(g['D']['primo'],   miglior_terza({'B', 'E', 'F', 'I', 'J'}))
    m82 = p(g['G']['primo'],   miglior_terza({'A', 'E', 'H', 'I', 'J'}))
    m83 = p(g['K']['secondo'], g['L']['secondo'])
    m84 = p(g['H']['primo'],   g['J']['secondo'])
    m85 = p(g['B']['primo'],   miglior_terza({'E', 'F', 'G', 'I', 'J'}))
    m86 = p(g['J']['primo'],   g['H']['secondo'])
    m87 = p(g['K']['primo'],   miglior_terza({'D', 'E', 'I', 'J', 'L'}))
    m88 = p(g['D']['secondo'], g['G']['secondo'])

    # --- Ottavi (match 89–96, 8 partite) ---
    m89  = p(m74, m77)
    m90  = p(m73, m75)
    m91  = p(m76, m78)
    m92  = p(m79, m80)
    m93  = p(m83, m84)
    m94  = p(m81, m82)
    m95  = p(m86, m88)
    m96  = p(m85, m87)

    # --- Quarti (match 97–100, 4 partite) ---
    m97  = p(m89, m90)
    m98  = p(m93, m94)
    m99  = p(m91, m92)
    m100 = p(m95, m96)

    # --- Semifinali (match 101–102, 2 partite) ---
    prob = simula_partita(m97, m98)
    finalista1, perdente1 = (m97, m98) if np.random.random() < prob else (m98, m97)

    prob = simula_partita(m99, m100)
    finalista2, perdente2 = (m99, m100) if np.random.random() < prob else (m100, m99)

    # --- Finale terzo posto (match 103, completa il conteggio di 104) ---
    p(perdente1, perdente2)

    # --- Finale (match 104) ---
    return p(finalista1, finalista2)['team']


MATCH_DESC = {
    73:  "2o Gio.A vs 2o Gio.B",          74:  "1o Gio.E vs 3a(A/B/C/D/F)",
    75:  "1o Gio.F vs 2o Gio.C",          76:  "1o Gio.C vs 2o Gio.F",
    77:  "1o Gio.I vs 3a(C/D/F/G/H)",     78:  "2o Gio.E vs 2o Gio.I",
    79:  "1o Gio.A vs 3a(C/E/F/H/I)",     80:  "1o Gio.L vs 3a(E/H/I/J/K)",
    81:  "1o Gio.D vs 3a(B/E/F/I/J)",     82:  "1o Gio.G vs 3a(A/E/H/I/J)",
    83:  "2o Gio.K vs 2o Gio.L",          84:  "1o Gio.H vs 2o Gio.J",
    85:  "1o Gio.B vs 3a(E/F/G/I/J)",     86:  "1o Gio.J vs 2o Gio.H",
    87:  "1o Gio.K vs 3a(D/E/I/J/L)",     88:  "2o Gio.D vs 2o Gio.G",
    89:  "Vin.M74 vs Vin.M77",            90:  "Vin.M73 vs Vin.M75",
    91:  "Vin.M76 vs Vin.M78",            92:  "Vin.M79 vs Vin.M80",
    93:  "Vin.M83 vs Vin.M84",            94:  "Vin.M81 vs Vin.M82",
    95:  "Vin.M86 vs Vin.M88",            96:  "Vin.M85 vs Vin.M87",
    97:  "Vin.M89 vs Vin.M90",            98:  "Vin.M93 vs Vin.M94",
    99:  "Vin.M91 vs Vin.M92",            100: "Vin.M95 vs Vin.M96",
    101: "Vin.M97 vs Vin.M98",            102: "Vin.M99 vs Vin.M100",
    103: "Perd.M101 vs Perd.M102",        104: "Vin.M101 vs Vin.M102",
}


def stampa_pronostici(df):
    n_sim = 1000
    SEP = "-" * 60

    # Strutture per la fase a gironi
    gironi_righe = {nome: [row for _, row in gr.iterrows()]
                    for nome, gr in df.groupby('girone')}

    stats_g = {
        nome: {r['team']: {'punti': 0.0, 'gf': 0.0, 'gs': 0.0, 'pass': 0}
               for r in righe}
        for nome, righe in gironi_righe.items()
    }

    stats_pm = {
        nome: {(i, j): {'ta': righe[i]['team'], 'tb': righe[j]['team'],
                        'ga': 0.0, 'gb': 0.0}
               for i in range(4) for j in range(i + 1, 4)}
        for nome, righe in gironi_righe.items()
    }

    # pa = lato sinistro del match, pb = lato destro, vt = vincitori
    br = {m: {'pa': {}, 'pb': {}, 'vt': {}} for m in range(73, 105)}

    def _mt(pool, gironi_ok):
        cand = [(n, sq, pt) for n, (sq, pt) in pool.items() if n in gironi_ok]
        if not cand:
            cand = [(n, sq, pt) for n, (sq, pt) in pool.items()]
        best = max(cand, key=lambda x: (x[2], calcola_forza(x[1])))
        del pool[best[0]]
        return best[1]

    def _pm(a, b, mn):
        ta, tb = a['team'], b['team']
        br[mn]['pa'][ta] = br[mn]['pa'].get(ta, 0) + 1
        br[mn]['pb'][tb] = br[mn]['pb'].get(tb, 0) + 1
        won_a = np.random.random() < simula_partita(a, b)
        w, l = (a, b) if won_a else (b, a)
        br[mn]['vt'][w['team']] = br[mn]['vt'].get(w['team'], 0) + 1
        return w, l

    print(f"Calcolo pronostici ({n_sim} simulazioni)...")

    for _ in range(n_sim):
        risultati = {}

        for nome, righe in gironi_righe.items():
            punti = {r['team']: 0 for r in righe}
            gf    = {r['team']: 0 for r in righe}
            gs    = {r['team']: 0 for r in righe}

            for i in range(4):
                for j in range(i + 1, 4):
                    a, b   = righe[i], righe[j]
                    ga, gb = simula_gol(a), simula_gol(b)
                    stats_pm[nome][(i, j)]['ga'] += ga / n_sim
                    stats_pm[nome][(i, j)]['gb'] += gb / n_sim

                    prob_a = simula_partita(a, b)
                    delta  = abs(calcola_forza(a) - calcola_forza(b))
                    prob_p = max(0.10, 0.28 - 0.20 * delta)
                    r_val  = np.random.random()

                    if r_val < prob_a * (1 - prob_p):
                        punti[a['team']] += 3
                    elif r_val < prob_a * (1 - prob_p) + prob_p:
                        punti[a['team']] += 1
                        punti[b['team']] += 1
                    else:
                        punti[b['team']] += 3

                    gf[a['team']] += ga;  gs[a['team']] += gb
                    gf[b['team']] += gb;  gs[b['team']] += ga

            classifica = sorted(righe,
                                key=lambda s: (punti[s['team']], calcola_forza(s)),
                                reverse=True)
            risultati[nome] = {
                'primo': classifica[0], 'secondo': classifica[1],
                'terzo': classifica[2], 'punti_terzo': punti[classifica[2]['team']],
            }
            for r_entry in righe:
                t = r_entry['team']
                stats_g[nome][t]['punti'] += punti[t] / n_sim
                stats_g[nome][t]['gf']    += gf[t]    / n_sim
                stats_g[nome][t]['gs']    += gs[t]    / n_sim
            stats_g[nome][classifica[0]['team']]['pass'] += 1
            stats_g[nome][classifica[1]['team']]['pass'] += 1

        terze_ord = sorted(
            [(n, r['terzo'], r['punti_terzo']) for n, r in risultati.items()],
            key=lambda x: (x[2], calcola_forza(x[1])), reverse=True
        )[:8]
        pool = {n: (sq, pt) for n, sq, pt in terze_ord}
        for n, sq, _ in terze_ord:
            stats_g[n][sq['team']]['pass'] += 1   # le migliori 8 terze avanzano

        g = risultati

        # Sedicesimi
        m73, _ = _pm(g['A']['secondo'], g['B']['secondo'],                73)
        m74, _ = _pm(g['E']['primo'],   _mt(pool, {'A','B','C','D','F'}), 74)
        m75, _ = _pm(g['F']['primo'],   g['C']['secondo'],                75)
        m76, _ = _pm(g['C']['primo'],   g['F']['secondo'],                76)
        m77, _ = _pm(g['I']['primo'],   _mt(pool, {'C','D','F','G','H'}), 77)
        m78, _ = _pm(g['E']['secondo'], g['I']['secondo'],                78)
        m79, _ = _pm(g['A']['primo'],   _mt(pool, {'C','E','F','H','I'}), 79)
        m80, _ = _pm(g['L']['primo'],   _mt(pool, {'E','H','I','J','K'}), 80)
        m81, _ = _pm(g['D']['primo'],   _mt(pool, {'B','E','F','I','J'}), 81)
        m82, _ = _pm(g['G']['primo'],   _mt(pool, {'A','E','H','I','J'}), 82)
        m83, _ = _pm(g['K']['secondo'], g['L']['secondo'],                83)
        m84, _ = _pm(g['H']['primo'],   g['J']['secondo'],                84)
        m85, _ = _pm(g['B']['primo'],   _mt(pool, {'E','F','G','I','J'}), 85)
        m86, _ = _pm(g['J']['primo'],   g['H']['secondo'],                86)
        m87, _ = _pm(g['K']['primo'],   _mt(pool, {'D','E','I','J','L'}), 87)
        m88, _ = _pm(g['D']['secondo'], g['G']['secondo'],                88)

        # Ottavi
        m89, _  = _pm(m74, m77,  89);  m90, _  = _pm(m73, m75,  90)
        m91, _  = _pm(m76, m78,  91);  m92, _  = _pm(m79, m80,  92)
        m93, _  = _pm(m83, m84,  93);  m94, _  = _pm(m81, m82,  94)
        m95, _  = _pm(m86, m88,  95);  m96, _  = _pm(m85, m87,  96)

        # Quarti
        m97,  _ = _pm(m89, m90,  97);  m98,  _ = _pm(m93, m94,  98)
        m99,  _ = _pm(m91, m92,  99);  m100, _ = _pm(m95, m96, 100)

        # Semifinali, finale 3o posto, finale
        f1, p1 = _pm(m97,  m98,  101)
        f2, p2 = _pm(m99,  m100, 102)
        _pm(p1, p2, 103)
        _pm(f1, f2, 104)

    # ---- STAMPA FASE A GIRONI ----
    print(f"\n{'='*60}")
    print("  FASE A GIRONI")
    print(f"{'='*60}")

    for nome in sorted(stats_g.keys()):
        print(f"\n  Girone {nome}")
        print(f"  {SEP}")
        print(f"  {'':3}  {'Squadra':<26}  {'Pti':>5}  {'GF':>5}  {'GS':>5}  {'Pass%':>6}")
        print(f"  {SEP}")
        classifica_g = sorted(stats_g[nome].items(),
                               key=lambda x: x[1]['punti'], reverse=True)
        for pos, (team, s) in enumerate(classifica_g, 1):
            mk = '*' if pos <= 2 else ' '
            print(f"  {pos}{mk}   {team:<26}  {s['punti']:>5.1f}  {s['gf']:>5.1f}"
                  f"  {s['gs']:>5.1f}  {s['pass']/n_sim:>5.0%}")
        print(f"\n    Risultati medi:")
        for (i, j), pm_d in sorted(stats_pm[nome].items()):
            print(f"      {pm_d['ta']:<26} {pm_d['ga']:.1f} - {pm_d['gb']:.1f}"
                  f"  {pm_d['tb']}")

    # ---- STAMPA FASE A ELIMINAZIONE DIRETTA ----
    print(f"\n{'='*60}")
    print("  FASE A ELIMINAZIONE DIRETTA")
    print(f"{'='*60}")

    turni = [
        ("SEDICESIMI",       range(73, 89)),
        ("OTTAVI DI FINALE", range(89, 97)),
        ("QUARTI DI FINALE", range(97, 101)),
        ("SEMIFINALI",       range(101, 103)),
        ("FINALE 3o POSTO",  [103]),
        ("FINALE",           [104]),
    ]

    for nome_turno, match_range in turni:
        print(f"\n  --- {nome_turno} ---")
        for mn in match_range:
            b = br[mn]
            if not b['pa'] or not b['pb']:
                continue
            t1, c1 = max(b['pa'].items(), key=lambda x: x[1])
            t2, c2 = max(b['pb'].items(), key=lambda x: x[1])
            v1 = b['vt'].get(t1, 0)
            v2 = b['vt'].get(t2, 0)
            vin = t1 if v1 >= v2 else t2
            print(f"  M{mn:<4} [{MATCH_DESC.get(mn, '')}]")
            print(f"         {t1:<24} ({c1/n_sim:>3.0%})  vs  {t2:<24} ({c2/n_sim:>3.0%})")
            print(f"         -> Pronostico: {vin}  ({max(v1, v2)/n_sim:.0%})\n")

    return stats_g, stats_pm, br, n_sim


def esporta_json(stats_g, stats_pm, br, vittorie, n_sim, df):
    # Abbinamento (i,j) -> giornata nel girone a 4 squadre
    giornata_map = {(0,1):1, (2,3):1, (0,2):2, (1,3):2, (0,3):3, (1,2):3}

    prob_vittoria = {t: round(v / n_sim * 100, 1) for t, v in vittorie.items() if v > 0}
    vincitore_finale = max(prob_vittoria.items(), key=lambda x: x[1])[0]

    gironi_json = {}
    for nome in sorted(stats_g.keys()):
        classifica_g = sorted(stats_g[nome].items(),
                               key=lambda x: x[1]['punti'], reverse=True)
        classifica_list = [
            {"pos": pos, "team": team,
             "punti": round(s['punti'], 1),
             "gf":    round(s['gf'],    1),
             "gs":    round(s['gs'],    1),
             "pass_pct": round(s['pass'] / n_sim * 100, 1)}
            for pos, (team, s) in enumerate(classifica_g, 1)
        ]
        partite_list = [
            {"giornata": giornata_map.get((i, j), 0),
             "squadra_a": pm_d['ta'], "gol_a": round(pm_d['ga'], 1),
             "gol_b":     round(pm_d['gb'], 1), "squadra_b": pm_d['tb']}
            for (i, j), pm_d in sorted(stats_pm[nome].items())
        ]
        gironi_json[nome] = {"classifica": classifica_list, "partite": partite_list}

    team_rows = {row['team']: row for _, row in df.iterrows()}
    n_gol = 1000

    bracket_json = {}
    for mn in range(73, 105):
        b = br[mn]
        if not b['pa'] or not b['pb']:
            continue
        t1, c1 = max(b['pa'].items(), key=lambda x: x[1])
        t2, c2 = max(b['pb'].items(), key=lambda x: x[1])
        v1 = b['vt'].get(t1, 0)
        v2 = b['vt'].get(t2, 0)
        vin = t1 if v1 >= v2 else t2

        # M103: mostra le due squadre eliminate in semifinale (non le finaliste)
        if mn == 103 and '101' in bracket_json and '102' in bracket_json:
            e101, e102 = bracket_json['101'], bracket_json['102']
            vin101, vin102 = e101['vincitore']['team'], e102['vincitore']['team']
            t1 = (e101['squadra_b']['team'] if e101['squadra_a']['team'] == vin101
                  else e101['squadra_a']['team'])
            t2 = (e102['squadra_b']['team'] if e102['squadra_a']['team'] == vin102
                  else e102['squadra_a']['team'])
            c1 = b['pa'].get(t1, 1)
            c2 = b['pb'].get(t2, 1)
            v1 = b['vt'].get(t1, 0)
            v2 = b['vt'].get(t2, 0)
            vin = t1 if v1 >= v2 else t2

        # Gol medi simulati con simula_gol() su n_gol iterazioni
        r1, r2 = team_rows.get(t1), team_rows.get(t2)
        if r1 is not None and r2 is not None:
            gol_a = round(float(np.mean([simula_gol(r1) for _ in range(n_gol)])), 1)
            gol_b = round(float(np.mean([simula_gol(r2) for _ in range(n_gol)])), 1)
        else:
            gol_a, gol_b = 0.0, 0.0

        bracket_json[str(mn)] = {
            "desc": MATCH_DESC.get(mn, ""),
            "squadra_a": {"team": t1, "prob_arrivo": round(c1 / n_sim * 100, 1)},
            "squadra_b": {"team": t2, "prob_arrivo": round(c2 / n_sim * 100, 1)},
            "vincitore": {"team": vin, "prob": round(max(v1, v2) / n_sim * 100, 1)},
            "gol_a": gol_a,
            "gol_b": gol_b,
        }

    output = {
        "vincitore_probabilita": prob_vittoria,
        "vincitore_finale":      vincitore_finale,
        "gironi":                gironi_json,
        "bracket":               bracket_json,
    }
    with open('data/results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print("Risultati esportati in data/results.json")


def main():
    df = carica_squadre()
    stats_g, stats_pm, br, n_sim_g = stampa_pronostici(df)

    vittorie = {team: 0 for team in df['team']}
    n_sim = 1000

    print(f"\nAvvio {n_sim} simulazioni per le probabilita' di vittoria...\n")
    for i in range(1, n_sim + 1):
        vincitore = simula_torneo(df)
        vittorie[vincitore] += 1
        if i % 100 == 0:
            print(f"  {i}/{n_sim} simulazioni completate")

    classifica = sorted(vittorie.items(), key=lambda x: x[1], reverse=True)

    sep = "-" * 42
    print(f"\n{sep}")
    print(f"  PROBABILITA' DI VITTORIA - Mondiale 2026")
    print(sep)
    print(f"  {'Squadra':<22} {'Prob':>7}")
    print(sep)
    for team, vitt in classifica:
        if vitt > 0:
            print(f"  {team:<22} {vitt / n_sim:>6.1%}")
    print(sep)

    esporta_json(stats_g, stats_pm, br, vittorie, n_sim_g, df)


if __name__ == '__main__':
    main()
