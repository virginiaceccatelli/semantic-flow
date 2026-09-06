#!/bin/tcsh
# E19 binding-language panel using the existing StarCoder2 J/R lenses.
# Run from the cluster checkout; the shared runner validates, resumes stage
# 206, and regenerates stage 205 without fitting new lenses.
setenv MODEL starcoder2-3b
exec tcsh jobs/lens_concepts.csh
