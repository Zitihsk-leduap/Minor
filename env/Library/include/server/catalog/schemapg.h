/*-------------------------------------------------------------------------
 *
 * schemapg.h
 *    Schema_pg_xxx macros for use by relcache.c
 *
 * Portions Copyright (c) 1996-2024, PostgreSQL Global Development Group
 * Portions Copyright (c) 1994, Regents of the University of California
 *
 * NOTES
 *  ******************************
 *  *** DO NOT EDIT THIS FILE! ***
 *  ******************************
 *
 *  It has been GENERATED by src/backend/catalog/genbki.pl
 *
 *-------------------------------------------------------------------------
 */
#ifndef SCHEMAPG_H
#define SCHEMAPG_H

#define Schema_pg_proc \
{ 1255, {"oid"}, 26, 4, 1, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1255, {"proname"}, 19, NAMEDATALEN, 2, -1, -1, 0, false, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 950 }, \
{ 1255, {"pronamespace"}, 26, 4, 3, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1255, {"proowner"}, 26, 4, 4, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1255, {"prolang"}, 26, 4, 5, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1255, {"procost"}, 700, 4, 6, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1255, {"prorows"}, 700, 4, 7, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1255, {"provariadic"}, 26, 4, 8, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1255, {"prosupport"}, 24, 4, 9, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1255, {"prokind"}, 18, 1, 10, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1255, {"prosecdef"}, 16, 1, 11, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1255, {"proleakproof"}, 16, 1, 12, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1255, {"proisstrict"}, 16, 1, 13, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1255, {"proretset"}, 16, 1, 14, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1255, {"provolatile"}, 18, 1, 15, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1255, {"proparallel"}, 18, 1, 16, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1255, {"pronargs"}, 21, 2, 17, -1, -1, 0, true, 's', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1255, {"pronargdefaults"}, 21, 2, 18, -1, -1, 0, true, 's', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1255, {"prorettype"}, 26, 4, 19, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1255, {"proargtypes"}, 30, -1, 20, -1, -1, 1, false, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1255, {"proallargtypes"}, 1028, -1, 21, -1, -1, 1, false, 'i', 'x', '\0', false, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1255, {"proargmodes"}, 1002, -1, 22, -1, -1, 1, false, 'i', 'x', '\0', false, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1255, {"proargnames"}, 1009, -1, 23, -1, -1, 1, false, 'i', 'x', '\0', false, false, false, '\0', '\0', false, true, 0, 950 }, \
{ 1255, {"proargdefaults"}, 194, -1, 24, -1, -1, 0, false, 'i', 'x', '\0', false, false, false, '\0', '\0', false, true, 0, 950 }, \
{ 1255, {"protrftypes"}, 1028, -1, 25, -1, -1, 1, false, 'i', 'x', '\0', false, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1255, {"prosrc"}, 25, -1, 26, -1, -1, 0, false, 'i', 'x', '\0', true, false, false, '\0', '\0', false, true, 0, 950 }, \
{ 1255, {"probin"}, 25, -1, 27, -1, -1, 0, false, 'i', 'x', '\0', false, false, false, '\0', '\0', false, true, 0, 950 }, \
{ 1255, {"prosqlbody"}, 194, -1, 28, -1, -1, 0, false, 'i', 'x', '\0', false, false, false, '\0', '\0', false, true, 0, 950 }, \
{ 1255, {"proconfig"}, 1009, -1, 29, -1, -1, 1, false, 'i', 'x', '\0', false, false, false, '\0', '\0', false, true, 0, 950 }, \
{ 1255, {"proacl"}, 1034, -1, 30, -1, -1, 1, false, 'd', 'x', '\0', false, false, false, '\0', '\0', false, true, 0, 0 }

#define Schema_pg_type \
{ 1247, {"oid"}, 26, 4, 1, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1247, {"typname"}, 19, NAMEDATALEN, 2, -1, -1, 0, false, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 950 }, \
{ 1247, {"typnamespace"}, 26, 4, 3, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1247, {"typowner"}, 26, 4, 4, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1247, {"typlen"}, 21, 2, 5, -1, -1, 0, true, 's', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1247, {"typbyval"}, 16, 1, 6, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1247, {"typtype"}, 18, 1, 7, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1247, {"typcategory"}, 18, 1, 8, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1247, {"typispreferred"}, 16, 1, 9, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1247, {"typisdefined"}, 16, 1, 10, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1247, {"typdelim"}, 18, 1, 11, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1247, {"typrelid"}, 26, 4, 12, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1247, {"typsubscript"}, 24, 4, 13, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1247, {"typelem"}, 26, 4, 14, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1247, {"typarray"}, 26, 4, 15, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1247, {"typinput"}, 24, 4, 16, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1247, {"typoutput"}, 24, 4, 17, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1247, {"typreceive"}, 24, 4, 18, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1247, {"typsend"}, 24, 4, 19, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1247, {"typmodin"}, 24, 4, 20, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1247, {"typmodout"}, 24, 4, 21, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1247, {"typanalyze"}, 24, 4, 22, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1247, {"typalign"}, 18, 1, 23, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1247, {"typstorage"}, 18, 1, 24, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1247, {"typnotnull"}, 16, 1, 25, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1247, {"typbasetype"}, 26, 4, 26, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1247, {"typtypmod"}, 23, 4, 27, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1247, {"typndims"}, 23, 4, 28, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1247, {"typcollation"}, 26, 4, 29, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1247, {"typdefaultbin"}, 194, -1, 30, -1, -1, 0, false, 'i', 'x', '\0', false, false, false, '\0', '\0', false, true, 0, 950 }, \
{ 1247, {"typdefault"}, 25, -1, 31, -1, -1, 0, false, 'i', 'x', '\0', false, false, false, '\0', '\0', false, true, 0, 950 }, \
{ 1247, {"typacl"}, 1034, -1, 32, -1, -1, 1, false, 'd', 'x', '\0', false, false, false, '\0', '\0', false, true, 0, 0 }

#define Schema_pg_attribute \
{ 1249, {"attrelid"}, 26, 4, 1, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1249, {"attname"}, 19, NAMEDATALEN, 2, -1, -1, 0, false, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 950 }, \
{ 1249, {"atttypid"}, 26, 4, 3, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1249, {"attlen"}, 21, 2, 4, -1, -1, 0, true, 's', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1249, {"attnum"}, 21, 2, 5, -1, -1, 0, true, 's', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1249, {"attcacheoff"}, 23, 4, 6, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1249, {"atttypmod"}, 23, 4, 7, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1249, {"attndims"}, 21, 2, 8, -1, -1, 0, true, 's', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1249, {"attbyval"}, 16, 1, 9, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1249, {"attalign"}, 18, 1, 10, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1249, {"attstorage"}, 18, 1, 11, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1249, {"attcompression"}, 18, 1, 12, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1249, {"attnotnull"}, 16, 1, 13, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1249, {"atthasdef"}, 16, 1, 14, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1249, {"atthasmissing"}, 16, 1, 15, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1249, {"attidentity"}, 18, 1, 16, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1249, {"attgenerated"}, 18, 1, 17, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1249, {"attisdropped"}, 16, 1, 18, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1249, {"attislocal"}, 16, 1, 19, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1249, {"attinhcount"}, 21, 2, 20, -1, -1, 0, true, 's', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1249, {"attcollation"}, 26, 4, 21, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1249, {"attstattarget"}, 21, 2, 22, -1, -1, 0, true, 's', 'p', '\0', false, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1249, {"attacl"}, 1034, -1, 23, -1, -1, 1, false, 'd', 'x', '\0', false, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1249, {"attoptions"}, 1009, -1, 24, -1, -1, 1, false, 'i', 'x', '\0', false, false, false, '\0', '\0', false, true, 0, 950 }, \
{ 1249, {"attfdwoptions"}, 1009, -1, 25, -1, -1, 1, false, 'i', 'x', '\0', false, false, false, '\0', '\0', false, true, 0, 950 }, \
{ 1249, {"attmissingval"}, 2277, -1, 26, -1, -1, 0, false, 'd', 'x', '\0', false, false, false, '\0', '\0', false, true, 0, 0 }

#define Schema_pg_class \
{ 1259, {"oid"}, 26, 4, 1, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1259, {"relname"}, 19, NAMEDATALEN, 2, -1, -1, 0, false, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 950 }, \
{ 1259, {"relnamespace"}, 26, 4, 3, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1259, {"reltype"}, 26, 4, 4, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1259, {"reloftype"}, 26, 4, 5, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1259, {"relowner"}, 26, 4, 6, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1259, {"relam"}, 26, 4, 7, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1259, {"relfilenode"}, 26, 4, 8, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1259, {"reltablespace"}, 26, 4, 9, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1259, {"relpages"}, 23, 4, 10, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1259, {"reltuples"}, 700, 4, 11, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1259, {"relallvisible"}, 23, 4, 12, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1259, {"reltoastrelid"}, 26, 4, 13, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1259, {"relhasindex"}, 16, 1, 14, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1259, {"relisshared"}, 16, 1, 15, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1259, {"relpersistence"}, 18, 1, 16, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1259, {"relkind"}, 18, 1, 17, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1259, {"relnatts"}, 21, 2, 18, -1, -1, 0, true, 's', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1259, {"relchecks"}, 21, 2, 19, -1, -1, 0, true, 's', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1259, {"relhasrules"}, 16, 1, 20, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1259, {"relhastriggers"}, 16, 1, 21, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1259, {"relhassubclass"}, 16, 1, 22, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1259, {"relrowsecurity"}, 16, 1, 23, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1259, {"relforcerowsecurity"}, 16, 1, 24, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1259, {"relispopulated"}, 16, 1, 25, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1259, {"relreplident"}, 18, 1, 26, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1259, {"relispartition"}, 16, 1, 27, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1259, {"relrewrite"}, 26, 4, 28, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1259, {"relfrozenxid"}, 28, 4, 29, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1259, {"relminmxid"}, 28, 4, 30, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1259, {"relacl"}, 1034, -1, 31, -1, -1, 1, false, 'd', 'x', '\0', false, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1259, {"reloptions"}, 1009, -1, 32, -1, -1, 1, false, 'i', 'x', '\0', false, false, false, '\0', '\0', false, true, 0, 950 }, \
{ 1259, {"relpartbound"}, 194, -1, 33, -1, -1, 0, false, 'i', 'x', '\0', false, false, false, '\0', '\0', false, true, 0, 950 }

#define Schema_pg_index \
{ 2610, {"indexrelid"}, 26, 4, 1, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 2610, {"indrelid"}, 26, 4, 2, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 2610, {"indnatts"}, 21, 2, 3, -1, -1, 0, true, 's', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 2610, {"indnkeyatts"}, 21, 2, 4, -1, -1, 0, true, 's', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 2610, {"indisunique"}, 16, 1, 5, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 2610, {"indnullsnotdistinct"}, 16, 1, 6, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 2610, {"indisprimary"}, 16, 1, 7, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 2610, {"indisexclusion"}, 16, 1, 8, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 2610, {"indimmediate"}, 16, 1, 9, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 2610, {"indisclustered"}, 16, 1, 10, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 2610, {"indisvalid"}, 16, 1, 11, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 2610, {"indcheckxmin"}, 16, 1, 12, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 2610, {"indisready"}, 16, 1, 13, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 2610, {"indislive"}, 16, 1, 14, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 2610, {"indisreplident"}, 16, 1, 15, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 2610, {"indkey"}, 22, -1, 16, -1, -1, 1, false, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 2610, {"indcollation"}, 30, -1, 17, -1, -1, 1, false, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 2610, {"indclass"}, 30, -1, 18, -1, -1, 1, false, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 2610, {"indoption"}, 22, -1, 19, -1, -1, 1, false, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 2610, {"indexprs"}, 194, -1, 20, -1, -1, 0, false, 'i', 'x', '\0', false, false, false, '\0', '\0', false, true, 0, 950 }, \
{ 2610, {"indpred"}, 194, -1, 21, -1, -1, 0, false, 'i', 'x', '\0', false, false, false, '\0', '\0', false, true, 0, 950 }

#define Schema_pg_database \
{ 1262, {"oid"}, 26, 4, 1, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1262, {"datname"}, 19, NAMEDATALEN, 2, -1, -1, 0, false, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 950 }, \
{ 1262, {"datdba"}, 26, 4, 3, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1262, {"encoding"}, 23, 4, 4, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1262, {"datlocprovider"}, 18, 1, 5, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1262, {"datistemplate"}, 16, 1, 6, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1262, {"datallowconn"}, 16, 1, 7, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1262, {"dathasloginevt"}, 16, 1, 8, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1262, {"datconnlimit"}, 23, 4, 9, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1262, {"datfrozenxid"}, 28, 4, 10, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1262, {"datminmxid"}, 28, 4, 11, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1262, {"dattablespace"}, 26, 4, 12, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1262, {"datcollate"}, 25, -1, 13, -1, -1, 0, false, 'i', 'x', '\0', true, false, false, '\0', '\0', false, true, 0, 950 }, \
{ 1262, {"datctype"}, 25, -1, 14, -1, -1, 0, false, 'i', 'x', '\0', true, false, false, '\0', '\0', false, true, 0, 950 }, \
{ 1262, {"datlocale"}, 25, -1, 15, -1, -1, 0, false, 'i', 'x', '\0', false, false, false, '\0', '\0', false, true, 0, 950 }, \
{ 1262, {"daticurules"}, 25, -1, 16, -1, -1, 0, false, 'i', 'x', '\0', false, false, false, '\0', '\0', false, true, 0, 950 }, \
{ 1262, {"datcollversion"}, 25, -1, 17, -1, -1, 0, false, 'i', 'x', '\0', false, false, false, '\0', '\0', false, true, 0, 950 }, \
{ 1262, {"datacl"}, 1034, -1, 18, -1, -1, 1, false, 'd', 'x', '\0', false, false, false, '\0', '\0', false, true, 0, 0 }

#define Schema_pg_authid \
{ 1260, {"oid"}, 26, 4, 1, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1260, {"rolname"}, 19, NAMEDATALEN, 2, -1, -1, 0, false, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 950 }, \
{ 1260, {"rolsuper"}, 16, 1, 3, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1260, {"rolinherit"}, 16, 1, 4, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1260, {"rolcreaterole"}, 16, 1, 5, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1260, {"rolcreatedb"}, 16, 1, 6, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1260, {"rolcanlogin"}, 16, 1, 7, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1260, {"rolreplication"}, 16, 1, 8, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1260, {"rolbypassrls"}, 16, 1, 9, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1260, {"rolconnlimit"}, 23, 4, 10, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1260, {"rolpassword"}, 25, -1, 11, -1, -1, 0, false, 'i', 'x', '\0', false, false, false, '\0', '\0', false, true, 0, 950 }, \
{ 1260, {"rolvaliduntil"}, 1184, 8, 12, -1, -1, 0, FLOAT8PASSBYVAL, 'd', 'p', '\0', false, false, false, '\0', '\0', false, true, 0, 0 }

#define Schema_pg_auth_members \
{ 1261, {"oid"}, 26, 4, 1, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1261, {"roleid"}, 26, 4, 2, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1261, {"member"}, 26, 4, 3, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1261, {"grantor"}, 26, 4, 4, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1261, {"admin_option"}, 16, 1, 5, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1261, {"inherit_option"}, 16, 1, 6, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 1261, {"set_option"}, 16, 1, 7, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }

#define Schema_pg_shseclabel \
{ 3592, {"objoid"}, 26, 4, 1, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 3592, {"classoid"}, 26, 4, 2, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 3592, {"provider"}, 25, -1, 3, -1, -1, 0, false, 'i', 'x', '\0', true, false, false, '\0', '\0', false, true, 0, 950 }, \
{ 3592, {"label"}, 25, -1, 4, -1, -1, 0, false, 'i', 'x', '\0', true, false, false, '\0', '\0', false, true, 0, 950 }

#define Schema_pg_subscription \
{ 6100, {"oid"}, 26, 4, 1, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 6100, {"subdbid"}, 26, 4, 2, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 6100, {"subskiplsn"}, 3220, 8, 3, -1, -1, 0, FLOAT8PASSBYVAL, 'd', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 6100, {"subname"}, 19, NAMEDATALEN, 4, -1, -1, 0, false, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 950 }, \
{ 6100, {"subowner"}, 26, 4, 5, -1, -1, 0, true, 'i', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 6100, {"subenabled"}, 16, 1, 6, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 6100, {"subbinary"}, 16, 1, 7, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 6100, {"substream"}, 18, 1, 8, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 6100, {"subtwophasestate"}, 18, 1, 9, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 6100, {"subdisableonerr"}, 16, 1, 10, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 6100, {"subpasswordrequired"}, 16, 1, 11, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 6100, {"subrunasowner"}, 16, 1, 12, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 6100, {"subfailover"}, 16, 1, 13, -1, -1, 0, true, 'c', 'p', '\0', true, false, false, '\0', '\0', false, true, 0, 0 }, \
{ 6100, {"subconninfo"}, 25, -1, 14, -1, -1, 0, false, 'i', 'x', '\0', true, false, false, '\0', '\0', false, true, 0, 950 }, \
{ 6100, {"subslotname"}, 19, NAMEDATALEN, 15, -1, -1, 0, false, 'c', 'p', '\0', false, false, false, '\0', '\0', false, true, 0, 950 }, \
{ 6100, {"subsynccommit"}, 25, -1, 16, -1, -1, 0, false, 'i', 'x', '\0', true, false, false, '\0', '\0', false, true, 0, 950 }, \
{ 6100, {"subpublications"}, 1009, -1, 17, -1, -1, 1, false, 'i', 'x', '\0', true, false, false, '\0', '\0', false, true, 0, 950 }, \
{ 6100, {"suborigin"}, 25, -1, 18, -1, -1, 0, false, 'i', 'x', '\0', false, false, false, '\0', '\0', false, true, 0, 950 }

#endif							/* SCHEMAPG_H */
