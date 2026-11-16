"""Does a model population's spatial code look like any brain region's?

The literature's place-cell measurements (field counts, field layout, active
fractions, population similarity against distance, remapping) run on model
units and are scored against ranges transcribed from
``docs/LITERATURE_PLACE_FIELDS_POPULATION_SIMILARITY.md``.

``measures``      pure functions on rate maps, usable on recordings
``populations``   one adapter from a spec string to a population
``geometry``      tracks, loops, arenas and rings on the scaffold
``run``           computes every measure for every population (``run.sh`` for Slurm)
``scorecard``     scores results against ``literature_targets.json``
``figures``       the figures in ``docs/EXPERIMENTS_CODING_COMPARISON.md``
"""
