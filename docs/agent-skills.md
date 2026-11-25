# AI Agent Skills

Soprano ships with a set of [Agent Skills](https://agentskills.io/) in the repository's [`skills/`](https://github.com/CCP-NC/soprano/tree/master/skills) directory.

## What are Agent Skills?

If you have only used AI models in a web browser, an **AI coding agent** (such as Claude Code, OpenCode, Codex, Cursor, or GitHub Copilot) is the next step: it runs in your terminal or editor alongside your project files, so it can inspect your `.magres` or `.cif` files, write Python scripts, and run `soprano` commands directly for you.

General-purpose AI models already know Python and ASE, but without up-to-date guidance they often guess outdated Soprano flags, re-implement built-in CLI commands by hand, or mix up solid-state NMR conventions (such as chemical-shift referencing or SIMPSON vs. `mrsimulator` Euler angles).

**Agent Skills** solve this. Each skill is a small, plain-text folder of instructions and reference tables that teaches an AI coding agent how to use a specific part of Soprano accurately. Once installed, your agent reads a short summary of each skill and automatically loads the full instructions whenever your question matches it.

### Available skills

The pages below are generated automatically from Soprano's `skills/` directory so they always match what your agent sees:

```{include} _generated/skills_summary.md
```

## Installing the skills

First make sure you have an AI coding tool that supports [Agent Skills](https://agentskills.io/) (for example Claude Code, OpenCode, Codex, or Cursor) and Soprano available (via `pip install soprano[moyo]` or on demand with `uvx`).

The quickest way to install all Soprano skills is with the Skills CLI (requires [Node.js](https://nodejs.org/) for `npx`):

```bash
npx skills add CCP-NC/soprano
```

The command will ask which AI tool(s) and skills you want to install.

If you prefer not to use `npx`, you can install manually by copying or symlinking any subfolder from [`skills/`](https://github.com/CCP-NC/soprano/tree/master/skills) into your tool's skills directory (for example `~/.agents/skills/`, `~/.claude/skills/`, or `.opencode/skills/`).

## Using the skills

Once installed, you do not need to learn any special syntax. Open your AI coding agent in the folder containing your crystal structures or `.magres` files and describe your task in plain language. Your agent will pick the relevant Soprano skill automatically.

If you want to be sure a specific skill is used, you can name it in your request (for example, *"using the `soprano-nmr` skill, ..."*).

Here are a few examples of what you can ask:

- *"Extract referenced $^{13}\mathrm{C}$ and $^1\mathrm{H}$ chemical shifts from all `.magres` files in this folder using references `C:170` and `H:30`, average the $\mathrm{CH}_3$ groups, and save a merged CSV table."*
- *"Build a SIMPSON spin-system file from `alanine.magres` for the $^{13}\mathrm{C}$ and $^1\mathrm{H}$ sites, including C–H dipolar couplings."*
- *"Split* `crystal.cif` *into intact molecules across periodic boundaries."*

:::{tip}  
Treat the agent like a helpful colleague at the terminal: review commands and scripts before running them, and double-check scientific inputs—such as chemical-shift references, isotope selections, and symmetry tolerances—against your own system. You can usually ask for a plot or visualisation as an output - these are easiest to spot any mistakes in. :::
