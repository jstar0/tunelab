# Security policy

TuneLab evaluates Python callables supplied by the user. Treat objectives and parameter values as code, not as untrusted configuration. Run experiments in an isolated environment when the objective loads external files, starts processes, or accesses a network.

The package does not transmit experiment data. Reports may contain parameter values and objective error messages, so review `outputs/` before publishing it.

To report a security issue, open a private report through the project owner rather than posting credentials, proprietary data, or an exploit in a public issue.
