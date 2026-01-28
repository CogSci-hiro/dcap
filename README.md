# dcap

Internal skeleton repo. README to be written later.


    cli/
      __init__.py
      main.py                 # single entry point for `dcap`
      commands/
        bids_convert.py         # subcommand module: add_subparser() + run()

    bids/
      __init__.py
      config.py               # dataclasses for BIDS conversion config
      converter.py            # orchestration: discover -> load -> transform -> write
      heuristics.py           # source discovery + parsing rules
      io.py                   # raw/audio loading + small file fixes
      transforms.py           # channel mapping/types/montage
      events.py               # events creation (optional)
      sidecars.py             # JSON/TSV sidecars (optional)
      anat.py                 # write_anat + derivatives copying (optional)
      _resources.py    