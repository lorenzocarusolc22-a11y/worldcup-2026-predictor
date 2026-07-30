# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Progetto

Simulatore del Mondiale di calcio 2026 nel formato reale a **48 squadre** (16 gironi da 3). Usa un sistema a punteggi ponderati per stimare la forza delle squadre e simulazioni Monte Carlo per calcolare le probabilità di vittoria finale di ogni squadra.

## Stack

- Python 3, pandas, numpy
- Nessuna dipendenza aggiuntiva oltre alle librerie sopra

## Dati

I dati delle squadre si trovano in `data/teams.csv`.

## Output

Probabilità di vittoria finale per ogni squadra, prodotta dalla simulazione Monte Carlo.

## Convenzioni

- Solo funzioni, nessuna classe o OOP
- Commenti nel codice scritti in italiano
