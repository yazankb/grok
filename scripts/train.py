#!/usr/bin/env python

import grok
import os

# Use cwd as project root so paths have correct encoding (avoid __file__ encoding issues)
_project_root = os.getcwd()

parser = grok.training.add_args()
parser.set_defaults(
    logdir=os.environ.get("GROK_LOGDIR", os.path.join(_project_root, "logs")),
    datadir=os.path.join(_project_root, "data"),
)
hparams = parser.parse_args()
hparams.datadir = os.path.join(_project_root, hparams.datadir) if not os.path.isabs(hparams.datadir) else hparams.datadir
hparams.datadir = os.path.normpath(hparams.datadir)
hparams.logdir = os.path.join(_project_root, hparams.logdir) if not os.path.isabs(hparams.logdir) else hparams.logdir
hparams.logdir = os.path.normpath(hparams.logdir)


print(hparams)
print(grok.training.train(hparams))
