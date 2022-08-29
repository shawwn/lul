from ...c.runtime import *
import os as _os

@PP_
class PP(PP):
    DIRECTORY_SEP: str

    ENABLE_CHECKING = 1

    # # /* src/config.h.  Generated from config.in by configure.  */
    # # /* src/config.in.  Generated from configure.ac by autoheader.  */
    # #
    # # /* GNU Emacs site configuration template file.
    # #
    # # Copyright (C) 1988, 1993-1994, 1999-2002, 2004-2021
    # #   Free Software Foundation, Inc.
    # #
    # # This file is part of GNU Emacs.
    # #
    # # GNU Emacs is free software: you can redistribute it and/or modify
    # # it under the terms of the GNU General Public License as published by
    # # the Free Software Foundation, either version 3 of the License, or (at
    # # your option) any later version.
    # #
    # # GNU Emacs is distributed in the hope that it will be useful,
    # # but WITHOUT ANY WARRANTY; without even the implied warranty of
    # # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    # # GNU General Public License for more details.
    # #
    # # You should have received a copy of the GNU General Public License
    # # along with GNU Emacs.  If not, see <https://www.gnu.org/licenses/>.  */
    # #
    # #
    # # /* No code in Emacs #includes config.h twice, but some bits of code
    # #    intended to work with other packages as well (like gmalloc.c)
    # #    think they can include it as many times as they like.  */
    # # #ifndef EMACS_CONFIG_H
    # if c_ifndef("EMACS_CONFIG_H"):
    # #define EMACS_CONFIG_H
    EMACS_CONFIG_H = None
    #
    # /* Define if building universal (internal helper macro) */
    # /* #undef AC_APPLE_UNIVERSAL_BUILD */
    #
    # /* Define to use the convention that & in the full name stands for the login
    #    id. */
    # #define AMPERSAND_FULL_NAME 1
    AMPERSAND_FULL_NAME = 1
    #
    # /* Define to the number of bits in type 'ptrdiff_t'. */
    # /* #undef BITSIZEOF_PTRDIFF_T */
    #
    # /* Define to the number of bits in type 'sig_atomic_t'. */
    # /* #undef BITSIZEOF_SIG_ATOMIC_T */
    #
    # /* Define to the number of bits in type 'size_t'. */
    # /* #undef BITSIZEOF_SIZE_T */
    #
    # /* Define to the number of bits in type 'wchar_t'. */
    # /* #undef BITSIZEOF_WCHAR_T */
    #
    # /* Define to the number of bits in type 'wint_t'. */
    # /* #undef BITSIZEOF_WINT_T */
    #
    # /* Define if get_current_dir_name should not be used. */
    # /* #undef BROKEN_GET_CURRENT_DIR_NAME */
    #
    # /* Define on FreeBSD to work around an issue when reading from a PTY. */
    # /* #undef BROKEN_PTY_READ_AFTER_EAGAIN */
    #
    # /* Define to enable compile-time checks for the Lisp_Object data type. */
    # /* #undef CHECK_LISP_OBJECT_TYPE */
    #
    # /* Define this to check whether someone updated the portable dumper code after
    #    changing the layout of a structure that it uses. If you change one of these
    #    structures, check that the pdumper.c code is still valid, and update the
    #    pertinent hash in pdumper.c by manually copying the hash from the
    #    newly-generated dmpstruct.h. */
    # /* #undef CHECK_STRUCTS */
    #
    # /* Short copyright string for this version of Emacs. */
    # #define COPYRIGHT "Copyright (C) 2021 Free Software Foundation, Inc."
    COPYRIGHT = "Copyright (C 2021 Free Software Foundation, Inc."
    #
    # /* Define if the system is Cygwin. */
    # /* #undef CYGWIN */
    #
    # /* Define to 1 if using 'alloca.c'. */
    # /* #undef C_ALLOCA */
    #
    # /* Define if the system is Darwin. */
    # #define DARWIN_OS /**/
    DARWIN_OS = None
    #
    # /* Name of the default sound device. */
    # #define DEFAULT_SOUND_DEVICE "/dev/dsp"
    DEFAULT_SOUND_DEVICE = "/dev/dsp"
    #
    # /* Define to 1 for DGUX with <sys/dg_sys_info.h>. */
    # /* #undef DGUX */
    #
    # /* Character that separates directories in a file name. */
    # #define DIRECTORY_SEP '/'
    DIRECTORY_SEP = _os.path.sep
    #
    # /* the name of the file descriptor member of DIR */
    # /* #undef DIR_FD_MEMBER_NAME */
    #
    # #ifdef DIR_FD_MEMBER_NAME
    # # define DIR_TO_FD(Dir_p) ((Dir_p)->DIR_FD_MEMBER_NAME)
    # #else
    # # define DIR_TO_FD(Dir_p) -1
    # #endif
    #
    #
    # /* Define if process.c does not need to close a pty to make it a controlling
    #    terminal (it is already a controlling terminal of the subprocess, because
    #    we did ioctl TIOCSCTTY). */
    # #define DONT_REOPEN_PTY 1
    DONT_REOPEN_PTY = 1
    #
    # /* Define if the system is MS DOS or MS Windows. */
    # /* #undef DOS_NT */
    #
    # /* Define to 1 if // is a file system root distinct from /. */
    # /* #undef DOUBLE_SLASH_IS_DISTINCT_ROOT */
    #
    # /* Define to 1 if the system memory allocator is Doug Lea style, with malloc
    #    hooks and malloc_set_state. */
    # /* #undef DOUG_LEA_MALLOC */
    #
    # /* Define to 1 to enable w32 debug facilities. */
    # /* #undef EMACSDEBUG */
    #
    # /* Define to the canonical Emacs configuration name. */
    # #define EMACS_CONFIGURATION "aarch64-apple-darwin20.6.0"
    EMACS_CONFIGURATION = "aarch64-apple-darwin20.6.0"
    #
    # /* Summary of some of the main features enabled by configure. */
    # #define EMACS_CONFIG_FEATURES "ACL GNUTLS JSON LCMS2 LIBXML2 MODULES NOTIFY KQUEUE NS PDUMPER PNG SQLITE3 THREADS TOOLKIT_SCROLL_BARS WEBP ZLIB"
    EMACS_CONFIG_FEATURES = "ACL GNUTLS JSON LCMS2 LIBXML2 MODULES NOTIFY KQUEUE NS PDUMPER PNG SQLITE3 THREADS TOOLKIT_SCROLL_BARS WEBP ZLIB"
    #
    # /* Define to the options passed to configure. */
    # #define EMACS_CONFIG_OPTIONS "--prefix=/opt/install/emacs-head"
    EMACS_CONFIG_OPTIONS = "--prefix=/opt/install/emacs-head"
    #
    # /* Define to 1 if expensive run-time data type and consistency checks are
    #    enabled. */
    # /* #undef ENABLE_CHECKING */
    #
    # /* Define this to 1 if F_DUPFD behavior does not match POSIX */
    # /* #undef FCNTL_DUPFD_BUGGY */
    #
    # /* Letter to use in finding device name of first PTY, if PTYs are supported.
    #    */
    # /* #undef FIRST_PTY_LETTER */
    #
    # /* Define to nothing if C supports flexible array members, and to 1 if it does
    #    not. That way, with a declaration like 'struct s { int n; short
    #    d[FLEXIBLE_ARRAY_MEMBER]; };', the struct hack can be used with pre-C99
    #    compilers. Use 'FLEXSIZEOF (struct s, d, N * sizeof (short))' to calculate
    #    the size in bytes of such a struct containing an N-element array. */
    # #define FLEXIBLE_ARRAY_MEMBER /**/
    FLEXIBLE_ARRAY_MEMBER = None
    #
    # /* Without the following workaround, Emacs runs slowly on OS X 10.8.
    #    The workaround disables some useful run-time checking, so it
    #    should be conditional to the platforms with the performance bug.
    #    Perhaps Apple will fix this some day; also see m4/extern-inline.m4.  */
    # #if defined __APPLE__ && defined __GNUC__
    # # ifndef _DONT_USE_CTYPE_INLINE_
    # #  define _DONT_USE_CTYPE_INLINE_
    # # endif
    # # ifndef _FORTIFY_SOURCE
    # #  define _FORTIFY_SOURCE 0
    # # endif
    # #endif
    #
    #
    # /* Define to 1 if realpath() can malloc memory, always gives an absolute path,
    #    and handles a trailing slash correctly. */
    # /* #undef FUNC_REALPATH_NEARLY_WORKS */
    #
    # /* Define to 1 if realpath() can malloc memory, always gives an absolute path,
    #    and handles leading slashes and a trailing slash correctly. */
    # /* #undef FUNC_REALPATH_WORKS */
    #
    # /* Define to 1 if futimesat mishandles a NULL file name. */
    # /* #undef FUTIMESAT_NULL_BUG */
    #
    # /* Define to 1 if --enable-gcc-warnings. */
    # /* #undef GCC_LINT */
    #
    # /* Define this temporarily to hunt a bug. If defined, the size of strings is
    #    redundantly recorded in sdata structures so that it can be compared to the
    #    sizes recorded in Lisp strings. */
    # /* #undef GC_CHECK_STRING_BYTES */
    #
    # /* Define this to check the string free list. */
    # /* #undef GC_CHECK_STRING_FREE_LIST */
    #
    # /* Define this to check for short string overrun. */
    # /* #undef GC_CHECK_STRING_OVERRUN */
    #
    # /* Mark a secondary stack, like the register stack on the ia64. */
    # /* #undef GC_MARK_SECONDARY_STACK */
    #
    # /* Define if setjmp is known to save all registers relevant for conservative
    #    garbage collection in the jmp_buf. */
    # #define GC_SETJMP_WORKS 1
    GC_SETJMP_WORKS = 1
    #
    # /* Define to 1 to disable GTK+/GDK deprecation warnings. */
    # /* #undef GDK_DISABLE_DEPRECATION_WARNINGS */
    #
    # /* Define to the type of elements in the array set by `getgroups'. Usually
    #    this is either `int' or `gid_t'. */
    # #define GETGROUPS_T gid_t
    GETGROUPS_T = gid_t
    #
    # /* Define this to 1 if getgroups(0,NULL) does not return the number of groups.
    #    */
    # /* #undef GETGROUPS_ZERO_BUG */
    #
    # /* Define this to 'void' or 'struct timezone' to match the system's
    #    declaration of the second argument to gettimeofday. */
    # #define GETTIMEOFDAY_TIMEZONE void
    GETTIMEOFDAY_TIMEZONE = None
    #
    # /* Define to 1 if getxattr works with XATTR_NAME_POSIX_ACL_ACCESS and
    #    XATTR_NAME_POSIX_ACL_DEFAULT. */
    # /* #undef GETXATTR_WITH_POSIX_ACLS */
    #
    # /* Define to 1 to disable Glib deprecation warnings. */
    # /* #undef GLIB_DISABLE_DEPRECATION_WARNINGS */
    #
    # /* Define this to enable glyphs debugging code. */
    # /* #undef GLYPH_DEBUG */
    #
    # /* Define to a C preprocessor expression that evaluates to 1 or 0, depending
    #    whether the gnulib module canonicalize-lgpl shall be considered present. */
    # #define GNULIB_CANONICALIZE_LGPL 1
    GNULIB_CANONICALIZE_LGPL = 1
    #
    # /* Define to a C preprocessor expression that evaluates to 1 or 0, depending
    #    whether the gnulib module close-stream shall be considered present. */
    # #define GNULIB_CLOSE_STREAM 1
    GNULIB_CLOSE_STREAM = 1
    #
    # /* Define to a C preprocessor expression that evaluates to 1 or 0, depending
    #    whether the gnulib module faccessat shall be considered present. */
    # #define GNULIB_FACCESSAT 1
    GNULIB_FACCESSAT = 1
    #
    # /* Define to a C preprocessor expression that evaluates to 1 or 0, depending
    #    whether the gnulib module fdopendir shall be considered present. */
    # #define GNULIB_FDOPENDIR 1
    GNULIB_FDOPENDIR = 1
    #
    # /* Define to a C preprocessor expression that evaluates to 1 or 0, depending
    #    whether the gnulib module fscanf shall be considered present. */
    # #define GNULIB_FSCANF 1
    GNULIB_FSCANF = 1
    #
    # /* Define to a C preprocessor expression that evaluates to 1 or 0, depending
    #    whether the gnulib module mkostemp shall be considered present. */
    # #define GNULIB_MKOSTEMP 1
    GNULIB_MKOSTEMP = 1
    #
    # /* enable some gnulib portability checks */
    # /* #undef GNULIB_PORTCHECK */
    #
    # /* Enable compile-time and run-time bounds-checking, and some warnings,
    # 	 without upsetting glibc 2.15+. */
    #       #if (defined GNULIB_PORTCHECK && !defined _FORTIFY_SOURCE \
    # 	   && defined __OPTIMIZE__ && __OPTIMIZE__)
    #       # define _FORTIFY_SOURCE 2
    #       #endif
    #
    #
    # /* Define to 1 if printf and friends should be labeled with attribute
    #    "__gnu_printf__" instead of "__printf__" */
    # /* #undef GNULIB_PRINTF_ATTRIBUTE_FLAVOR_GNU */
    #
    # /* Define to a C preprocessor expression that evaluates to 1 or 0, depending
    #    whether the gnulib module scanf shall be considered present. */
    # #define GNULIB_SCANF 1
    GNULIB_SCANF = 1
    #
    # /* Define to 1 if you want the FILE stream functions getc, putc, etc. to use
    #    unlocked I/O if available, throughout the package. Unlocked I/O can improve
    #    performance, sometimes dramatically. But unlocked I/O is safe only in
    #    single-threaded programs, as well as in multithreaded programs for which
    #    you can guarantee that every FILE stream, including stdin, stdout, stderr,
    #    is used only in a single thread. */
    # #define GNULIB_STDIO_SINGLE_THREAD 1
    GNULIB_STDIO_SINGLE_THREAD = 1
    #
    # /* Define to a C preprocessor expression that evaluates to 1 or 0, depending
    #    whether the gnulib module tempname shall be considered present. */
    # #define GNULIB_TEMPNAME 1
    GNULIB_TEMPNAME = 1
    #
    # /* Define if ths system is compatible with GNU/Linux. */
    # /* #undef GNU_LINUX */
    #
    # /* Define to 1 if you want to use the GNU memory allocator. */
    # /* #undef GNU_MALLOC */
    #
    # /* Define if the system is Haiku. */
    # /* #undef HAIKU */
    #
    # /* Define to 1 if you have the `accept4' function. */
    # /* #undef HAVE_ACCEPT4 */
    #
    # /* Define to 1 if you have the `access' function. */
    # #define HAVE_ACCESS 1
    HAVE_ACCESS = 1
    #
    # /* Define to 1 if you have the `aclsort' function. */
    # /* #undef HAVE_ACLSORT */
    #
    # /* Define to 1 if you have the <aclv.h> header file. */
    # /* #undef HAVE_ACLV_H */
    #
    # /* Define to 1 if you have the `aclx_get' function. */
    # /* #undef HAVE_ACLX_GET */
    #
    # /* Define to 1 if you have the `acl_copy_ext_native' function. */
    # #define HAVE_ACL_COPY_EXT_NATIVE 1
    HAVE_ACL_COPY_EXT_NATIVE = 1
    #
    # /* Define to 1 if you have the `acl_create_entry_np' function. */
    # #define HAVE_ACL_CREATE_ENTRY_NP 1
    HAVE_ACL_CREATE_ENTRY_NP = 1
    #
    # /* Define to 1 if you have the `acl_delete_def_file' function. */
    # #define HAVE_ACL_DELETE_DEF_FILE 1
    HAVE_ACL_DELETE_DEF_FILE = 1
    #
    # /* Define to 1 if you have the `acl_delete_fd_np' function. */
    # #define HAVE_ACL_DELETE_FD_NP 1
    HAVE_ACL_DELETE_FD_NP = 1
    #
    # /* Define to 1 if you have the `acl_delete_file_np' function. */
    # #define HAVE_ACL_DELETE_FILE_NP 1
    HAVE_ACL_DELETE_FILE_NP = 1
    #
    # /* Define to 1 if you have the `acl_entries' function. */
    # /* #undef HAVE_ACL_ENTRIES */
    #
    # /* Define to 1 if you have the `acl_extended_file' function. */
    # /* #undef HAVE_ACL_EXTENDED_FILE */
    #
    # /* Define to 1 if the constant ACL_FIRST_ENTRY exists. */
    # #define HAVE_ACL_FIRST_ENTRY 1
    HAVE_ACL_FIRST_ENTRY = 1
    #
    # /* Define to 1 if you have the `acl_free' function. */
    # #define HAVE_ACL_FREE 1
    HAVE_ACL_FREE = 1
    #
    # /* Define to 1 if you have the `acl_free_text' function. */
    # /* #undef HAVE_ACL_FREE_TEXT */
    #
    # /* Define to 1 if you have the `acl_from_mode' function. */
    # /* #undef HAVE_ACL_FROM_MODE */
    #
    # /* Define to 1 if you have the `acl_from_text' function. */
    # #define HAVE_ACL_FROM_TEXT 1
    HAVE_ACL_FROM_TEXT = 1
    #
    # /* Define to 1 if you have the `acl_get_fd' function. */
    # #define HAVE_ACL_GET_FD 1
    HAVE_ACL_GET_FD = 1
    #
    # /* Define to 1 if you have the `acl_get_file' function. */
    # #define HAVE_ACL_GET_FILE 1
    HAVE_ACL_GET_FILE = 1
    #
    # /* Define to 1 if you have the <acl/libacl.h> header file. */
    # /* #undef HAVE_ACL_LIBACL_H */
    #
    # /* Define to 1 if you have the `acl_set_fd' function. */
    # #define HAVE_ACL_SET_FD 1
    HAVE_ACL_SET_FD = 1
    #
    # /* Define to 1 if you have the `acl_set_file' function. */
    # #define HAVE_ACL_SET_FILE 1
    HAVE_ACL_SET_FILE = 1
    #
    # /* Define to 1 if you have the `acl_to_short_text' function. */
    # /* #undef HAVE_ACL_TO_SHORT_TEXT */
    #
    # /* Define to 1 if you have the `acl_trivial' function. */
    # /* #undef HAVE_ACL_TRIVIAL */
    #
    # /* Define to 1 if the ACL type ACL_TYPE_EXTENDED exists. */
    # #define HAVE_ACL_TYPE_EXTENDED 1
    HAVE_ACL_TYPE_EXTENDED = 1
    #
    # /* Define to 1 if you have the `aligned_alloc' function. */
    # #define HAVE_ALIGNED_ALLOC 1
    HAVE_ALIGNED_ALLOC = 1
    #
    # /* Define to 1 if you have 'alloca' after including <alloca.h>, a header that
    #    may be supplied by this distribution. */
    # #define HAVE_ALLOCA 1
    HAVE_ALLOCA = 1
    #
    # /* Define to 1 if <alloca.h> works. */
    # #define HAVE_ALLOCA_H 1
    HAVE_ALLOCA_H = 1
    #
    # /* Define to 1 if ALSA is available. */
    # /* #undef HAVE_ALSA */
    #
    # /* Define to 1 if you have the <bcrypt.h> header file. */
    # /* #undef HAVE_BCRYPT_H */
    #
    # /* Define to 1 if you have the <byteswap.h> header file. */
    # /* #undef HAVE_BYTESWAP_H */
    #
    # /* Define to 1 if you have the `canonicalize_file_name' function. */
    # /* #undef HAVE_CANONICALIZE_FILE_NAME */
    #
    # /* Define to 1 if you have the `cfmakeraw' function. */
    # #define HAVE_CFMAKERAW 1
    HAVE_CFMAKERAW = 1
    #
    # /* Define to 1 if you have the `cfsetspeed' function. */
    # #define HAVE_CFSETSPEED 1
    HAVE_CFSETSPEED = 1
    #
    # /* Define to 1 if you have the `clock_gettime' function. */
    # #define HAVE_CLOCK_GETTIME 1
    HAVE_CLOCK_GETTIME = 1
    #
    # /* Define to 1 if you have the `clock_settime' function. */
    # #define HAVE_CLOCK_SETTIME 1
    HAVE_CLOCK_SETTIME = 1
    #
    # /* Define to 1 if you have the <coff.h> header file. */
    # /* #undef HAVE_COFF_H */
    #
    # /* Define to 1 if you have the <com_err.h> header file. */
    # /* #undef HAVE_COM_ERR_H */
    #
    # /* Define to 1 if opening a FIFO, socket, or symlink with O_PATH is buggy. */
    # /* #undef HAVE_CYGWIN_O_PATH_BUG */
    #
    # /* Define to 1 if C supports variable-length arrays. */
    # #define HAVE_C_VARARRAYS 1
    HAVE_C_VARARRAYS = 1
    #
    # /* Define to 1 if data_start is the address of the start of the main data
    #    segment. */
    # /* #undef HAVE_DATA_START */
    #
    # /* Define to 1 if using D-Bus. */
    # /* #undef HAVE_DBUS */
    #
    # /* Define to 1 if you have the `dbus_type_is_valid' function. */
    # /* #undef HAVE_DBUS_TYPE_IS_VALID */
    #
    # /* Define to 1 if you have the `dbus_validate_bus_name' function. */
    # /* #undef HAVE_DBUS_VALIDATE_BUS_NAME */
    #
    # /* Define to 1 if you have the `dbus_validate_interface' function. */
    # /* #undef HAVE_DBUS_VALIDATE_INTERFACE */
    #
    # /* Define to 1 if you have the `dbus_validate_member' function. */
    # /* #undef HAVE_DBUS_VALIDATE_MEMBER */
    #
    # /* Define to 1 if you have the `dbus_validate_path' function. */
    # /* #undef HAVE_DBUS_VALIDATE_PATH */
    #
    # /* Define to 1 if you have the `dbus_watch_get_unix_fd' function. */
    # /* #undef HAVE_DBUS_WATCH_GET_UNIX_FD */
    #
    # /* Define to 1 if you have the declaration of `alarm', and to 0 if you don't.
    #    */
    # #define HAVE_DECL_ALARM 1
    HAVE_DECL_ALARM = 1
    #
    # /* Define to 1 if you have the declaration of `aligned_alloc', and to 0 if you
    #    don't. */
    # #define HAVE_DECL_ALIGNED_ALLOC 1
    HAVE_DECL_ALIGNED_ALLOC = 1
    #
    # /* Define to 1 if you have the declaration of `clearerr_unlocked', and to 0 if
    #    you don't. */
    # #define HAVE_DECL_CLEARERR_UNLOCKED 1
    HAVE_DECL_CLEARERR_UNLOCKED = 1
    #
    # /* Define to 1 if you have the declaration of `dirfd', and to 0 if you don't.
    #    */
    # /* #undef HAVE_DECL_DIRFD */
    #
    # /* Define to 1 if you have the declaration of `ecvt', and to 0 if you don't.
    #    */
    # #define HAVE_DECL_ECVT 1
    HAVE_DECL_ECVT = 1
    #
    # /* Define to 1 if you have the declaration of `execvpe', and to 0 if you
    #    don't. */
    # #define HAVE_DECL_EXECVPE 0
    HAVE_DECL_EXECVPE = 0
    #
    # /* Define to 1 if you have the declaration of `fcloseall', and to 0 if you
    #    don't. */
    # #define HAVE_DECL_FCLOSEALL 0
    HAVE_DECL_FCLOSEALL = 0
    #
    # /* Define to 1 if you have the declaration of `fcvt', and to 0 if you don't.
    #    */
    # #define HAVE_DECL_FCVT 1
    HAVE_DECL_FCVT = 1
    #
    # /* Define to 1 if you have the declaration of `fdopendir', and to 0 if you
    #    don't. */
    # #define HAVE_DECL_FDOPENDIR 1
    HAVE_DECL_FDOPENDIR = 1
    #
    # /* Define to 1 if you have the declaration of `feof_unlocked', and to 0 if you
    #    don't. */
    # #define HAVE_DECL_FEOF_UNLOCKED 1
    HAVE_DECL_FEOF_UNLOCKED = 1
    #
    # /* Define to 1 if you have the declaration of `ferror_unlocked', and to 0 if
    #    you don't. */
    # #define HAVE_DECL_FERROR_UNLOCKED 1
    HAVE_DECL_FERROR_UNLOCKED = 1
    #
    # /* Define to 1 if you have the declaration of `fflush_unlocked', and to 0 if
    #    you don't. */
    # #define HAVE_DECL_FFLUSH_UNLOCKED 0
    HAVE_DECL_FFLUSH_UNLOCKED = 0
    #
    # /* Define to 1 if you have the declaration of `fgets_unlocked', and to 0 if
    #    you don't. */
    # #define HAVE_DECL_FGETS_UNLOCKED 0
    HAVE_DECL_FGETS_UNLOCKED = 0
    #
    # /* Define to 1 if you have the declaration of `fputc_unlocked', and to 0 if
    #    you don't. */
    # #define HAVE_DECL_FPUTC_UNLOCKED 0
    HAVE_DECL_FPUTC_UNLOCKED = 0
    #
    # /* Define to 1 if you have the declaration of `fputs_unlocked', and to 0 if
    #    you don't. */
    # #define HAVE_DECL_FPUTS_UNLOCKED 0
    HAVE_DECL_FPUTS_UNLOCKED = 0
    #
    # /* Define to 1 if you have the declaration of `fread_unlocked', and to 0 if
    #    you don't. */
    # #define HAVE_DECL_FREAD_UNLOCKED 0
    HAVE_DECL_FREAD_UNLOCKED = 0
    #
    # /* Define to 1 if you have the declaration of `fwrite_unlocked', and to 0 if
    #    you don't. */
    # #define HAVE_DECL_FWRITE_UNLOCKED 0
    HAVE_DECL_FWRITE_UNLOCKED = 0
    #
    # /* Define to 1 if you have the declaration of `gcvt', and to 0 if you don't.
    #    */
    # #define HAVE_DECL_GCVT 1
    HAVE_DECL_GCVT = 1
    #
    # /* Define to 1 if you have the declaration of `getchar_unlocked', and to 0 if
    #    you don't. */
    # #define HAVE_DECL_GETCHAR_UNLOCKED 1
    HAVE_DECL_GETCHAR_UNLOCKED = 1
    #
    # /* Define to 1 if you have the declaration of `getc_unlocked', and to 0 if you
    #    don't. */
    # #define HAVE_DECL_GETC_UNLOCKED 1
    HAVE_DECL_GETC_UNLOCKED = 1
    #
    # /* Define to 1 if you have the declaration of `getdtablesize', and to 0 if you
    #    don't. */
    # #define HAVE_DECL_GETDTABLESIZE 1
    HAVE_DECL_GETDTABLESIZE = 1
    #
    # /* Define to 1 if you have the declaration of `isblank', and to 0 if you
    #    don't. */
    # #define HAVE_DECL_ISBLANK 1
    HAVE_DECL_ISBLANK = 1
    #
    # /* Define to 1 if you have the declaration of `localtime_r', and to 0 if you
    #    don't. */
    # #define HAVE_DECL_LOCALTIME_R 1
    HAVE_DECL_LOCALTIME_R = 1
    #
    # /* Define to 1 if you have the declaration of `memmem', and to 0 if you don't.
    #    */
    # #define HAVE_DECL_MEMMEM 1
    HAVE_DECL_MEMMEM = 1
    #
    # /* Define to 1 if you have the declaration of `memrchr', and to 0 if you
    #    don't. */
    # #define HAVE_DECL_MEMRCHR 0
    HAVE_DECL_MEMRCHR = 0
    #
    # /* Define to 1 if you have the declaration of `POSIX_SPAWN_SETSID', and to 0
    #    if you don't. */
    # #define HAVE_DECL_POSIX_SPAWN_SETSID 1
    HAVE_DECL_POSIX_SPAWN_SETSID = 1
    #
    # /* Define to 1 if you have the declaration of `putchar_unlocked', and to 0 if
    #    you don't. */
    # #define HAVE_DECL_PUTCHAR_UNLOCKED 1
    HAVE_DECL_PUTCHAR_UNLOCKED = 1
    #
    # /* Define to 1 if you have the declaration of `putc_unlocked', and to 0 if you
    #    don't. */
    # #define HAVE_DECL_PUTC_UNLOCKED 1
    HAVE_DECL_PUTC_UNLOCKED = 1
    #
    # /* Define to 1 if you have the declaration of `SECCOMP_FILTER_FLAG_TSYNC', and
    #    to 0 if you don't. */
    # /* #undef HAVE_DECL_SECCOMP_FILTER_FLAG_TSYNC */
    #
    # /* Define to 1 if you have the declaration of `SECCOMP_SET_MODE_FILTER', and
    #    to 0 if you don't. */
    # /* #undef HAVE_DECL_SECCOMP_SET_MODE_FILTER */
    #
    # /* Define to 1 if you have the declaration of `strmode', and to 0 if you
    #    don't. */
    # #define HAVE_DECL_STRMODE 1
    HAVE_DECL_STRMODE = 1
    #
    # /* Define to 1 if you have the declaration of `strnlen', and to 0 if you
    #    don't. */
    # #define HAVE_DECL_STRNLEN 1
    HAVE_DECL_STRNLEN = 1
    #
    # /* Define to 1 if you have the declaration of `strtoimax', and to 0 if you
    #    don't. */
    # #define HAVE_DECL_STRTOIMAX 1
    HAVE_DECL_STRTOIMAX = 1
    #
    # /* Define to 1 if you have the declaration of `strtoll', and to 0 if you
    #    don't. */
    # /* #undef HAVE_DECL_STRTOLL */
    #
    # /* Define to 1 if you have the declaration of `tzname', and to 0 if you don't.
    #    */
    # /* #undef HAVE_DECL_TZNAME */
    #
    # /* Define to 1 if you have the declaration of `__fpending', and to 0 if you
    #    don't. */
    # /* #undef HAVE_DECL___FPENDING */
    #
    # /* Define to 1 if you have the <dirent.h> header file. */
    # #define HAVE_DIRENT_H 1
    HAVE_DIRENT_H = 1
    #
    # /* Define to 1 if you have the `dirfd' function. */
    # /* #undef HAVE_DIRFD */
    #
    # /* Define to 1 if you have the `dladdr' function. */
    # #define HAVE_DLADDR 1
    HAVE_DLADDR = 1
    #
    # /* Define to 1 if you have the `dlfunc' function. */
    # /* #undef HAVE_DLFUNC */
    #
    # /* Define to 1 if you have the `eaccess' function. */
    # /* #undef HAVE_EACCESS */
    #
    # /* Define to 1 if you have the `endgrent' function. */
    # #define HAVE_ENDGRENT 1
    HAVE_ENDGRENT = 1
    #
    # /* Define to 1 if you have the `endpwent' function. */
    # #define HAVE_ENDPWENT 1
    HAVE_ENDPWENT = 1
    #
    # /* Define if you have the declaration of environ. */
    # /* #undef HAVE_ENVIRON_DECL */
    #
    # /* Define to 1 if you have the `euidaccess' function. */
    # /* #undef HAVE_EUIDACCESS */
    #
    # /* Define to 1 if you have the <execinfo.h> header file. */
    # #define HAVE_EXECINFO_H 1
    HAVE_EXECINFO_H = 1
    #
    # /* Define to 1 if you have the `explicit_bzero' function. */
    # /* #undef HAVE_EXPLICIT_BZERO */
    #
    # /* Define to 1 if you have the `explicit_memset' function. */
    # /* #undef HAVE_EXPLICIT_MEMSET */
    #
    # /* Define to 1 if you have the `faccessat' function. */
    # #define HAVE_FACCESSAT 1
    HAVE_FACCESSAT = 1
    #
    # /* Define to 1 if you have the `facl' function. */
    # /* #undef HAVE_FACL */
    #
    # /* Define to 1 if you have the `fchdir' function. */
    # #define HAVE_FCHDIR 1
    HAVE_FCHDIR = 1
    #
    # /* Define to 1 if you have the `fchmod' function. */
    # #define HAVE_FCHMOD 1
    HAVE_FCHMOD = 1
    #
    # /* Define to 1 if you have the `fchmodat' function. */
    # #define HAVE_FCHMODAT 1
    HAVE_FCHMODAT = 1
    #
    # /* Define to 1 if you have the `fcntl' function. */
    # #define HAVE_FCNTL 1
    HAVE_FCNTL = 1
    #
    # /* Define to 1 if you have the `fdopendir' function. */
    # #define HAVE_FDOPENDIR 1
    HAVE_FDOPENDIR = 1
    #
    # /* Define to 1 if you have the `fork' function. */
    # #define HAVE_FORK 1
    HAVE_FORK = 1
    #
    # /* Define to 1 if you have the `freeifaddrs' function. */
    # #define HAVE_FREEIFADDRS 1
    HAVE_FREEIFADDRS = 1
    #
    # /* Define to 1 if using the freetype and fontconfig libraries. */
    # /* #undef HAVE_FREETYPE */
    #
    # /* Define if the 'free' function is guaranteed to preserve errno. */
    # /* #undef HAVE_FREE_POSIX */
    #
    # /* Define to 1 if fseeko (and presumably ftello) exists and is declared. */
    # #define HAVE_FSEEKO 1
    HAVE_FSEEKO = 1
    #
    # /* Define to 1 if you have the `fstatat' function. */
    # #define HAVE_FSTATAT 1
    HAVE_FSTATAT = 1
    #
    # /* Define to 1 if you have the `fsync' function. */
    # #define HAVE_FSYNC 1
    HAVE_FSYNC = 1
    #
    # /* Define to 1 if you have the `FT_Face_GetCharVariantIndex' function. */
    # /* #undef HAVE_FT_FACE_GETCHARVARIANTINDEX */
    #
    # /* Define to 1 if you have the `futimens' function. */
    # #define HAVE_FUTIMENS 1
    HAVE_FUTIMENS = 1
    #
    # /* Define to 1 if you have the `futimes' function. */
    # #define HAVE_FUTIMES 1
    HAVE_FUTIMES = 1
    #
    # /* Define to 1 if you have the `futimesat' function. */
    # /* #undef HAVE_FUTIMESAT */
    #
    # /* Define to 1 if you have the `gai_strerror' function. */
    # #define HAVE_GAI_STRERROR 1
    HAVE_GAI_STRERROR = 1
    #
    # /* Define to 1 if using GConf. */
    # /* #undef HAVE_GCONF */
    #
    # /* Define to 1 if you have the `getacl' function. */
    # /* #undef HAVE_GETACL */
    #
    # /* Define to 1 if you have getaddrinfo_a for asynchronous DNS resolution. */
    # /* #undef HAVE_GETADDRINFO_A */
    #
    # /* Define to 1 if you have the `getcwd' function. */
    # #define HAVE_GETCWD 1
    HAVE_GETCWD = 1
    #
    # /* Define to 1 if you have the `getdtablesize' function. */
    # #define HAVE_GETDTABLESIZE 1
    HAVE_GETDTABLESIZE = 1
    #
    # /* Define to 1 if you have the `getgrent' function. */
    # #define HAVE_GETGRENT 1
    HAVE_GETGRENT = 1
    #
    # /* Define to 1 if your system has a working `getgroups' function. */
    # #define HAVE_GETGROUPS 1
    HAVE_GETGROUPS = 1
    #
    # /* Define to 1 if you have the `gethostname' function. */
    # #define HAVE_GETHOSTNAME 1
    HAVE_GETHOSTNAME = 1
    #
    # /* Define to 1 if you have the `getifaddrs' function. */
    # #define HAVE_GETIFADDRS 1
    HAVE_GETIFADDRS = 1
    #
    # /* Define to 1 if you have the <getopt.h> header file. */
    # #define HAVE_GETOPT_H 1
    HAVE_GETOPT_H = 1
    #
    # /* Define to 1 if you have the `getopt_long_only' function. */
    # #define HAVE_GETOPT_LONG_ONLY 1
    HAVE_GETOPT_LONG_ONLY = 1
    #
    # /* Define to 1 if you have the `getpagesize' function. */
    # #define HAVE_GETPAGESIZE 1
    HAVE_GETPAGESIZE = 1
    #
    # /* Define to 1 if you have the `getpt' function. */
    # /* #undef HAVE_GETPT */
    #
    # /* Define to 1 if you have the `getpwent' function. */
    # #define HAVE_GETPWENT 1
    HAVE_GETPWENT = 1
    #
    # /* Define to 1 if you have the `getrandom' function. */
    # /* #undef HAVE_GETRANDOM */
    #
    # /* Define to 1 if you have the `getrlimit' function. */
    # #define HAVE_GETRLIMIT 1
    HAVE_GETRLIMIT = 1
    #
    # /* Define to 1 if you have the `getrusage' function. */
    # #define HAVE_GETRUSAGE 1
    HAVE_GETRUSAGE = 1
    #
    # /* Define to 1 if you have the `getsockname' function. */
    # #define HAVE_GETSOCKNAME 1
    HAVE_GETSOCKNAME = 1
    #
    # /* Define to 1 if you have the `gettimeofday' function. */
    # #define HAVE_GETTIMEOFDAY 1
    HAVE_GETTIMEOFDAY = 1
    #
    # /* Define to 1 if you have the `get_current_dir_name' function. */
    # /* #undef HAVE_GET_CURRENT_DIR_NAME */
    #
    # /* Define to 1 if using GFile. */
    # /* #undef HAVE_GFILENOTIFY */
    #
    # /* Define to 1 if you have a gif (or ungif) library. */
    # /* #undef HAVE_GIF */
    #
    # /* Define to 1 if GLib is linked in. */
    # /* #undef HAVE_GLIB */
    #
    # /* Define to 1 if you have the <gmp/gmp.h> header file. */
    # /* #undef HAVE_GMP_GMP_H */
    #
    # /* Define to 1 if you have the <gmp.h> header file. */
    # /* #undef HAVE_GMP_H */
    #
    # /* Define if using GnuTLS. */
    # #define HAVE_GNUTLS 1
    HAVE_GNUTLS = 1
    #
    # /* Define to 1 if you have the gpm library (-lgpm). */
    # /* #undef HAVE_GPM */
    #
    # /* Define to 1 if you have the `grantpt' function. */
    # #define HAVE_GRANTPT 1
    HAVE_GRANTPT = 1
    #
    # /* Define to 1 if using GSettings. */
    # /* #undef HAVE_GSETTINGS */
    #
    # /* Define to 1 if using GTK 3 or later. */
    # /* #undef HAVE_GTK3 */
    #
    # /* Define to 1 if you have the `gtk_file_selection_new' function. */
    # /* #undef HAVE_GTK_FILE_SELECTION_NEW */
    #
    # /* Define to 1 if you have the `gtk_window_set_has_resize_grip' function. */
    # /* #undef HAVE_GTK_WINDOW_SET_HAS_RESIZE_GRIP */
    #
    # /* Define if Emacs will be built with Haiku windowing support */
    # /* #undef HAVE_HAIKU */
    #
    # /* Define to 1 if using HarfBuzz. */
    # /* #undef HAVE_HARFBUZZ */
    #
    # /* Define to 1 if you have the <ieee754.h> header file. */
    # /* #undef HAVE_IEEE754_H */
    #
    # /* Define to 1 if you have the <ifaddrs.h> header file. */
    # #define HAVE_IFADDRS_H 1
    HAVE_IFADDRS_H = 1
    #
    # /* Define to 1 if using ImageMagick. */
    # /* #undef HAVE_IMAGEMAGICK */
    #
    # /* Define to 1 if using ImageMagick7. */
    # /* #undef HAVE_IMAGEMAGICK7 */
    #
    # /* Define to 1 to use inotify. */
    # /* #undef HAVE_INOTIFY */
    #
    # /* Define to 1 if you have the <inttypes.h> header file. */
    # #define HAVE_INTTYPES_H 1
    HAVE_INTTYPES_H = 1
    #
    # /* Define to 1 if you have the `isblank' function. */
    # #define HAVE_ISBLANK 1
    HAVE_ISBLANK = 1
    #
    # /* Define to 1 if you have the `iswctype' function. */
    # #define HAVE_ISWCTYPE 1
    HAVE_ISWCTYPE = 1
    #
    # /* Define to 1 if you have the jpeg library (typically -ljpeg). */
    # /* #undef HAVE_JPEG */
    #
    # /* Define if using Jansson. */
    # #define HAVE_JSON 1
    HAVE_JSON = 1
    #
    # /* Define to 1 if you have the <kerberosIV/krb.h> header file. */
    # /* #undef HAVE_KERBEROSIV_KRB_H */
    #
    # /* Define to 1 if you have the <kerberos/krb.h> header file. */
    # /* #undef HAVE_KERBEROS_KRB_H */
    #
    # /* Define to 1 to use kqueue. */
    # #define HAVE_KQUEUE 1
    HAVE_KQUEUE = 1
    #
    # /* Define to 1 if `e_text' is a member of `krb5_error'. */
    # /* #undef HAVE_KRB5_ERROR_E_TEXT */
    #
    # /* Define to 1 if `text' is a member of `krb5_error'. */
    # /* #undef HAVE_KRB5_ERROR_TEXT */
    #
    # /* Define to 1 if you have the <krb5.h> header file. */
    # /* #undef HAVE_KRB5_H */
    #
    # /* Define to 1 if you have the <krb.h> header file. */
    # /* #undef HAVE_KRB_H */
    #
    # /* Define if you have <langinfo.h> and nl_langinfo (CODESET). */
    # #define HAVE_LANGINFO_CODESET 1
    HAVE_LANGINFO_CODESET = 1
    #
    # /* Define if you have <langinfo.h> and nl_langinfo (_NL_PAPER_WIDTH). */
    # /* #undef HAVE_LANGINFO__NL_PAPER_WIDTH */
    #
    # /* Define to 1 if you have the `lchmod' function. */
    # #define HAVE_LCHMOD 1
    HAVE_LCHMOD = 1
    #
    # /* Define to 1 if you have the lcms2 library (-llcms2). */
    # #define HAVE_LCMS2 1
    HAVE_LCMS2 = 1
    #
    # /* Define to 1 if you have the `dgc' library (-ldgc). */
    # /* #undef HAVE_LIBDGC */
    #
    # /* Define to 1 if you have the `gccjit' library (-lgccjit). */
    # /* #undef HAVE_LIBGCCJIT */
    #
    # /* Define to 1 if you have the <libgccjit.h> header file. */
    # /* #undef HAVE_LIBGCCJIT_H */
    #
    # /* Define to 1 if you have the <libgen.h> header file. */
    # #define HAVE_LIBGEN_H 1
    HAVE_LIBGEN_H = 1
    #
    # /* Define to 1 if you have the <libintl.h> header file. */
    # /* #undef HAVE_LIBINTL_H */
    #
    # /* Define to 1 if you have the `kstat' library (-lkstat). */
    # /* #undef HAVE_LIBKSTAT */
    #
    # /* Define to 1 if you have the 'lockfile' library (-llockfile). */
    # /* #undef HAVE_LIBLOCKFILE */
    #
    # /* Define to 1 if you have the 'mail' library (-lmail). */
    # /* #undef HAVE_LIBMAIL */
    #
    # /* Define to 1 if using libotf. */
    # /* #undef HAVE_LIBOTF */
    #
    # /* Define to 1 if you have the `perfstat' library (-lperfstat). */
    # /* #undef HAVE_LIBPERFSTAT */
    #
    # /* Define to 1 if using SELinux. */
    # /* #undef HAVE_LIBSELINUX */
    #
    # /* Define if using libsystemd. */
    # /* #undef HAVE_LIBSYSTEMD */
    #
    # /* Define to 1 if you have the libxml library (-lxml2). */
    # #define HAVE_LIBXML2 1
    HAVE_LIBXML2 = 1
    #
    # /* Define to 1 if the bcrypt library is guaranteed to be present. */
    # /* #undef HAVE_LIB_BCRYPT */
    #
    # /* Define to 1 if you have the <limits.h> header file. */
    # #define HAVE_LIMITS_H 1
    HAVE_LIMITS_H = 1
    #
    # /* Define to 1 if you have the <linux/filter.h> header file. */
    # /* #undef HAVE_LINUX_FILTER_H */
    #
    # /* Define to 1 if you have the <linux/fs.h> header file. */
    # /* #undef HAVE_LINUX_FS_H */
    #
    # /* Define to 1 if you have the <linux/seccomp.h> header file. */
    # /* #undef HAVE_LINUX_SECCOMP_H */
    #
    # /* Define to 1 if you have Linux sysinfo function. */
    # /* #undef HAVE_LINUX_SYSINFO */
    #
    # /* Define if localtime-like functions can loop forever on extreme arguments.
    #    */
    # /* #undef HAVE_LOCALTIME_INFLOOP_BUG */
    #
    # /* Define to 1 if you have the `localtime_r' function. */
    # #define HAVE_LOCALTIME_R 1
    HAVE_LOCALTIME_R = 1
    #
    # /* Define to 1 if you have the `log2' function. */
    # #define HAVE_LOG2 1
    HAVE_LOG2 = 1
    #
    # /* Define to 1 if the system has the type 'long long int'. */
    # #define HAVE_LONG_LONG_INT 1
    HAVE_LONG_LONG_INT = 1
    #
    # /* Define to 1 if you have the `lrand48' function. */
    # #define HAVE_LRAND48 1
    HAVE_LRAND48 = 1
    #
    # /* Define to 1 if you have the `lstat' function. */
    # #define HAVE_LSTAT 1
    HAVE_LSTAT = 1
    #
    # /* Define to 1 if you have the `lutimes' function. */
    # #define HAVE_LUTIMES 1
    HAVE_LUTIMES = 1
    #
    # /* Define to 1 if using libm17n-flt. */
    # /* #undef HAVE_M17N_FLT */
    #
    # /* Define to 1 if you have the <machine/soundcard.h> header file. */
    # /* #undef HAVE_MACHINE_SOUNDCARD_H */
    #
    # /* Define to 1 if you have the <mach/mach.h> header file. */
    # /* #undef HAVE_MACH_MACH_H */
    #
    # /* Define to 1 if you have the `MagickAutoOrientImage' function. */
    # /* #undef HAVE_MAGICKAUTOORIENTIMAGE */
    #
    # /* Define to 1 if you have the `MagickExportImagePixels' function. */
    # /* #undef HAVE_MAGICKEXPORTIMAGEPIXELS */
    #
    # /* Define to 1 if you have the `MagickMergeImageLayers' function. */
    # /* #undef HAVE_MAGICKMERGEIMAGELAYERS */
    #
    # /* Define to 1 if you have the `MagickRelinquishMemory' function. */
    # /* #undef HAVE_MAGICKRELINQUISHMEMORY */
    #
    # /* Define to 1 if you have the <maillock.h> header file. */
    # /* #undef HAVE_MAILLOCK_H */
    #
    # /* Define to 1 if you have the <malloc.h> header file. */
    # /* #undef HAVE_MALLOC_H */
    #
    # /* Define to 1 if you have the <malloc/malloc.h> header file. */
    # #define HAVE_MALLOC_MALLOC_H 1
    HAVE_MALLOC_MALLOC_H = 1
    #
    # /* Define if malloc, realloc, and calloc set errno on allocation failure. */
    # #define HAVE_MALLOC_POSIX 1
    HAVE_MALLOC_POSIX = 1
    #
    # /* Define to 1 if <wchar.h> declares mbstate_t. */
    # #define HAVE_MBSTATE_T 1
    HAVE_MBSTATE_T = 1
    #
    # /* Define to 1 if you have the `memmem' function. */
    # #define HAVE_MEMMEM 1
    HAVE_MEMMEM = 1
    #
    # /* Define to 1 if you have the `mempcpy' function. */
    # /* #undef HAVE_MEMPCPY */
    #
    # /* Define to 1 if you have the `memrchr' function. */
    # /* #undef HAVE_MEMRCHR */
    #
    # /* Define to 1 if you have the `memset_s' function. */
    # #define HAVE_MEMSET_S 1
    HAVE_MEMSET_S = 1
    #
    # /* Define to 1 if you have the <minix/config.h> header file. */
    # /* #undef HAVE_MINIX_CONFIG_H */
    #
    # /* Define to 1 if <limits.h> defines the MIN and MAX macros. */
    # /* #undef HAVE_MINMAX_IN_LIMITS_H */
    #
    # /* Define to 1 if <sys/param.h> defines the MIN and MAX macros. */
    # #define HAVE_MINMAX_IN_SYS_PARAM_H 1
    HAVE_MINMAX_IN_SYS_PARAM_H = 1
    #
    # /* Define to 1 if you have the `mkostemp' function. */
    # #define HAVE_MKOSTEMP 1
    HAVE_MKOSTEMP = 1
    #
    # /* Define to 1 if you have a working `mmap' system call. */
    # #define HAVE_MMAP 1
    HAVE_MMAP = 1
    #
    # /* Define to 1 if you have the <mmsystem.h> header file. */
    # /* #undef HAVE_MMSYSTEM_H */
    #
    # /* Define to 1 if dynamic modules are enabled */
    # #define HAVE_MODULES 1
    HAVE_MODULES = 1
    #
    # /* Define to 1 if native compiler is available. */
    # /* #undef HAVE_NATIVE_COMP */
    #
    # /* Define to use native OS APIs for images. */
    # #define HAVE_NATIVE_IMAGE_API 1
    HAVE_NATIVE_IMAGE_API = 1
    #
    # /* Define to 1 if fchmodat works, except for the trailing slash handling. */
    # /* #undef HAVE_NEARLY_WORKING_FCHMODAT */
    #
    # /* Define to 1 if utimensat works, except for the trailing slash handling. */
    # /* #undef HAVE_NEARLY_WORKING_UTIMENSAT */
    #
    # /* Define to 1 if you have the <net/if_dl.h> header file. */
    # #define HAVE_NET_IF_DL_H 1
    HAVE_NET_IF_DL_H = 1
    #
    # /* Define to 1 if you have the <net/if.h> header file. */
    # #define HAVE_NET_IF_H 1
    HAVE_NET_IF_H = 1
    #
    # /* Define to 1 if you have the `newlocale' function. */
    # #define HAVE_NEWLOCALE 1
    HAVE_NEWLOCALE = 1
    #
    # /* Define to 1 if you have the <nlist.h> header file. */
    # /* #undef HAVE_NLIST_H */
    #
    # /* Define to 1 if you are using the NeXTstep API, either GNUstep or Cocoa on
    #    macOS. */
    # #define HAVE_NS 1
    HAVE_NS = 1
    #
    # /* Define to use native MS Windows GUI. */
    # /* #undef HAVE_NTGUI */
    #
    # /* Define to 1 if libotf has OTF_get_variation_glyphs. */
    # /* #undef HAVE_OTF_GET_VARIATION_GLYPHS */
    #
    # /* Define to 1 if libotf is affected by https://debbugs.gnu.org/28110. */
    # /* #undef HAVE_OTF_KANNADA_BUG */
    #
    # /* Define to build with portable dumper support */
    # #define HAVE_PDUMPER 1
    HAVE_PDUMPER = 1
    #
    # /* Define to 1 if personality flag ADDR_NO_RANDOMIZE exists. */
    # /* #undef HAVE_PERSONALITY_ADDR_NO_RANDOMIZE */
    #
    # /* Define to 1 if you have the `pipe2' function. */
    # /* #undef HAVE_PIPE2 */
    #
    # /* Define to 1 if you have the png library. */
    # #define HAVE_PNG 1
    HAVE_PNG = 1
    #
    # /* Define to 1 if you have the `posix_madvise' function. */
    # #define HAVE_POSIX_MADVISE 1
    HAVE_POSIX_MADVISE = 1
    #
    # /* Define to 1 if you have the `posix_memalign' function. */
    # /* #undef HAVE_POSIX_MEMALIGN */
    #
    # /* Define to 1 if you have the `posix_openpt' function. */
    # #define HAVE_POSIX_OPENPT 1
    HAVE_POSIX_OPENPT = 1
    #
    # /* Define to 1 if you have the `posix_spawn' function. */
    # #define HAVE_POSIX_SPAWN 1
    HAVE_POSIX_SPAWN = 1
    #
    # /* Define to 1 if you have the `posix_spawnattr_setflags' function. */
    # #define HAVE_POSIX_SPAWNATTR_SETFLAGS 1
    HAVE_POSIX_SPAWNATTR_SETFLAGS = 1
    #
    # /* Define to 1 if you have the `posix_spawn_file_actions_addchdir' function.
    #    */
    # /* #undef HAVE_POSIX_SPAWN_FILE_ACTIONS_ADDCHDIR */
    #
    # /* Define to 1 if you have the `posix_spawn_file_actions_addchdir_np'
    #    function. */
    # #define HAVE_POSIX_SPAWN_FILE_ACTIONS_ADDCHDIR_NP 1
    HAVE_POSIX_SPAWN_FILE_ACTIONS_ADDCHDIR_NP = 1
    #
    # /* Define if you have the /proc filesystem. */
    # /* #undef HAVE_PROCFS */
    #
    # /* Define to 1 if you have the `pselect' function. */
    # #define HAVE_PSELECT 1
    HAVE_PSELECT = 1
    #
    # /* Define to 1 if you have the `pstat_getdynamic' function. */
    # /* #undef HAVE_PSTAT_GETDYNAMIC */
    #
    # /* Define to 1 if you have POSIX threads. */
    # #define HAVE_PTHREAD 1
    HAVE_PTHREAD = 1
    #
    # /* Define to 1 if you have the <pthread.h> header file. */
    # #define HAVE_PTHREAD_H 1
    HAVE_PTHREAD_H = 1
    #
    # /* Define to 1 if you have the `pthread_setname_np' function. */
    # #define HAVE_PTHREAD_SETNAME_NP 1
    HAVE_PTHREAD_SETNAME_NP = 1
    #
    # /* Define to 1 if pthread_setname_np takes a single argument. */
    # #define HAVE_PTHREAD_SETNAME_NP_1ARG 1
    HAVE_PTHREAD_SETNAME_NP_1ARG = 1
    #
    # /* Define to 1 if pthread_setname_np takes three arguments. */
    # /* #undef HAVE_PTHREAD_SETNAME_NP_3ARG */
    #
    # /* Define to 1 if you have the `pthread_set_name_np' function. */
    # /* #undef HAVE_PTHREAD_SET_NAME_NP */
    #
    # /* Define to 1 if the pthread_sigmask function can be used (despite bugs). */
    # #define HAVE_PTHREAD_SIGMASK 1
    HAVE_PTHREAD_SIGMASK = 1
    #
    # /* Define if the system supports pty devices. */
    # #define HAVE_PTYS 1
    HAVE_PTYS = 1
    #
    # /* Define to 1 if you have the <pty.h> header file. */
    # /* #undef HAVE_PTY_H */
    #
    # /* Define to 1 if you have the <pwd.h> header file. */
    # #define HAVE_PWD_H 1
    HAVE_PWD_H = 1
    #
    # /* Define to 1 if you have the `random' function. */
    # #define HAVE_RANDOM 1
    HAVE_RANDOM = 1
    #
    # /* Define to 1 if you have the `rawmemchr' function. */
    # /* #undef HAVE_RAWMEMCHR */
    #
    # /* Define to 1 if you have the `readlink' function. */
    # #define HAVE_READLINK 1
    HAVE_READLINK = 1
    #
    # /* Define to 1 if you have the `readlinkat' function. */
    # #define HAVE_READLINKAT 1
    HAVE_READLINKAT = 1
    #
    # /* Define to 1 if you have the `realpath' function. */
    # #define HAVE_REALPATH 1
    HAVE_REALPATH = 1
    #
    # /* Define to 1 if you have the `recvfrom' function. */
    # #define HAVE_RECVFROM 1
    HAVE_RECVFROM = 1
    #
    # /* Define to 1 if you have the `rint' function. */
    # #define HAVE_RINT 1
    HAVE_RINT = 1
    #
    # /* Define to 1 if using librsvg. */
    # /* #undef HAVE_RSVG */
    #
    # /* Define to 1 if you have the <sanitizer/lsan_interface.h> header file. */
    # #define HAVE_SANITIZER_LSAN_INTERFACE_H 1
    HAVE_SANITIZER_LSAN_INTERFACE_H = 1
    #
    # /* Define to 1 if you have the `sbrk' function. */
    # #define HAVE_SBRK 1
    HAVE_SBRK = 1
    #
    # /* Define to 1 if you have the `sched_getaffinity' function. */
    # /* #undef HAVE_SCHED_GETAFFINITY */
    #
    # /* Define to 1 if sched_getaffinity has a glibc compatible declaration. */
    # /* #undef HAVE_SCHED_GETAFFINITY_LIKE_GLIBC */
    #
    # /* Define to 1 if you have the `sched_getaffinity_np' function. */
    # /* #undef HAVE_SCHED_GETAFFINITY_NP */
    #
    # /* Define to 1 if you have the `select' function. */
    # #define HAVE_SELECT 1
    HAVE_SELECT = 1
    #
    # /* Define to 1 if you have the `sendto' function. */
    # #define HAVE_SENDTO 1
    HAVE_SENDTO = 1
    #
    # /* Define to 1 if you have the `setdtablesize' function. */
    # /* #undef HAVE_SETDTABLESIZE */
    #
    # /* Define to 1 if you have the `setitimer' function. */
    # #define HAVE_SETITIMER 1
    HAVE_SETITIMER = 1
    #
    # /* Define to 1 if you have the `setlocale' function. */
    # #define HAVE_SETLOCALE 1
    HAVE_SETLOCALE = 1
    #
    # /* Define to 1 if you have the `setrlimit' function. */
    # #define HAVE_SETRLIMIT 1
    HAVE_SETRLIMIT = 1
    #
    # /* Define to 1 if you have the `shutdown' function. */
    # #define HAVE_SHUTDOWN 1
    HAVE_SHUTDOWN = 1
    #
    # /* Define to 1 if you have the `sig2str' function. */
    # /* #undef HAVE_SIG2STR */
    #
    # /* Define to 1 if you have the `sigdescr_np' function. */
    # /* #undef HAVE_SIGDESCR_NP */
    #
    # /* Define to 1 if 'sig_atomic_t' is a signed integer type. */
    # /* #undef HAVE_SIGNED_SIG_ATOMIC_T */
    #
    # /* Define to 1 if 'wchar_t' is a signed integer type. */
    # /* #undef HAVE_SIGNED_WCHAR_T */
    #
    # /* Define to 1 if 'wint_t' is a signed integer type. */
    # /* #undef HAVE_SIGNED_WINT_T */
    #
    # /* Define to 1 if sigsetjmp and siglongjmp work. */
    # #define HAVE_SIGSETJMP 1
    HAVE_SIGSETJMP = 1
    #
    # /* Define to 1 if the system has the type `sigset_t'. */
    # #define HAVE_SIGSET_T 1
    HAVE_SIGSET_T = 1
    #
    # /* Define to 1 if you have the `snprintf' function. */
    # #define HAVE_SNPRINTF 1
    HAVE_SNPRINTF = 1
    #
    # /* Define if the system supports 4.2-compatible sockets. */
    # #define HAVE_SOCKETS 1
    HAVE_SOCKETS = 1
    #
    # /* Define to 1 if you have sound support. */
    # /* #undef HAVE_SOUND */
    #
    # /* Define to 1 if you have the <soundcard.h> header file. */
    # /* #undef HAVE_SOUNDCARD_H */
    #
    # /* Define to 1 if you have the <spawn.h> header file. */
    # #define HAVE_SPAWN_H 1
    HAVE_SPAWN_H = 1
    #
    # /* Define to 1 if you have the libsqlite3 library (-lsqlite). */
    # #define HAVE_SQLITE3 1
    HAVE_SQLITE3 = 1
    #
    # /* Define to 1 if sqlite3 supports loading extensions. */
    # /* #undef HAVE_SQLITE3_LOAD_EXTENSION */
    #
    # /* Define to 1 if C stack overflow can be handled in some cases. */
    # /* #undef HAVE_STACK_OVERFLOW_HANDLING */
    #
    # /* Define to 1 if you have the `statacl' function. */
    # /* #undef HAVE_STATACL */
    #
    # /* Define to 1 if statement expressions work. */
    # #define HAVE_STATEMENT_EXPRESSIONS 1
    HAVE_STATEMENT_EXPRESSIONS = 1
    #
    # /* Define to 1 if you have the <stdint.h> header file. */
    # #define HAVE_STDINT_H 1
    HAVE_STDINT_H = 1
    #
    # /* Define to 1 if you have the <stdio_ext.h> header file. */
    # /* #undef HAVE_STDIO_EXT_H */
    #
    # /* Define to 1 if you have the <stdio.h> header file. */
    # #define HAVE_STDIO_H 1
    HAVE_STDIO_H = 1
    #
    # /* Define to 1 if you have the <stdlib.h> header file. */
    # #define HAVE_STDLIB_H 1
    HAVE_STDLIB_H = 1
    #
    # /* Define to 1 if you have the `stpcpy' function. */
    # #define HAVE_STPCPY 1
    HAVE_STPCPY = 1
    #
    # /* Define to 1 if you have the <strings.h> header file. */
    # #define HAVE_STRINGS_H 1
    HAVE_STRINGS_H = 1
    #
    # /* Define to 1 if you have the <string.h> header file. */
    # #define HAVE_STRING_H 1
    HAVE_STRING_H = 1
    #
    # /* Define to 1 if you have the `strsignal' function. */
    # #define HAVE_STRSIGNAL 1
    HAVE_STRSIGNAL = 1
    #
    # /* Define to 1 if you have the `strtoimax' function. */
    # #define HAVE_STRTOIMAX 1
    HAVE_STRTOIMAX = 1
    #
    # /* Define to 1 if you have the `strtoll' function. */
    # /* #undef HAVE_STRTOLL */
    #
    # /* Define to 1 if 'struct __attribute__ ((aligned (N)))' aligns the structure
    #    to an N-byte boundary. */
    # #define HAVE_STRUCT_ATTRIBUTE_ALIGNED 1
    HAVE_STRUCT_ATTRIBUTE_ALIGNED = 1
    #
    # /* Define if there is a member named d_type in the struct describing directory
    #    headers. */
    # #define HAVE_STRUCT_DIRENT_D_TYPE 1
    HAVE_STRUCT_DIRENT_D_TYPE = 1
    #
    # /* Define to 1 if `ifr_addr' is a member of `struct ifreq'. */
    # #define HAVE_STRUCT_IFREQ_IFR_ADDR 1
    HAVE_STRUCT_IFREQ_IFR_ADDR = 1
    #
    # /* Define to 1 if `ifr_addr.sa_len' is a member of `struct ifreq'. */
    # #define HAVE_STRUCT_IFREQ_IFR_ADDR_SA_LEN 1
    HAVE_STRUCT_IFREQ_IFR_ADDR_SA_LEN = 1
    #
    # /* Define to 1 if `ifr_broadaddr' is a member of `struct ifreq'. */
    # #define HAVE_STRUCT_IFREQ_IFR_BROADADDR 1
    HAVE_STRUCT_IFREQ_IFR_BROADADDR = 1
    #
    # /* Define to 1 if `ifr_flags' is a member of `struct ifreq'. */
    # #define HAVE_STRUCT_IFREQ_IFR_FLAGS 1
    HAVE_STRUCT_IFREQ_IFR_FLAGS = 1
    #
    # /* Define to 1 if `ifr_hwaddr' is a member of `struct ifreq'. */
    # /* #undef HAVE_STRUCT_IFREQ_IFR_HWADDR */
    #
    # /* Define to 1 if `ifr_netmask' is a member of `struct ifreq'. */
    # /* #undef HAVE_STRUCT_IFREQ_IFR_NETMASK */
    #
    # /* Define to 1 if `n_un.n_name' is a member of `struct nlist'. */
    # /* #undef HAVE_STRUCT_NLIST_N_UN_N_NAME */
    #
    # /* Define to 1 if `st_atimensec' is a member of `struct stat'. */
    # /* #undef HAVE_STRUCT_STAT_ST_ATIMENSEC */
    #
    # /* Define to 1 if `st_atimespec.tv_nsec' is a member of `struct stat'. */
    # #define HAVE_STRUCT_STAT_ST_ATIMESPEC_TV_NSEC 1
    HAVE_STRUCT_STAT_ST_ATIMESPEC_TV_NSEC = 1
    #
    # /* Define to 1 if `st_atim.st__tim.tv_nsec' is a member of `struct stat'. */
    # /* #undef HAVE_STRUCT_STAT_ST_ATIM_ST__TIM_TV_NSEC */
    #
    # /* Define to 1 if `st_atim.tv_nsec' is a member of `struct stat'. */
    # /* #undef HAVE_STRUCT_STAT_ST_ATIM_TV_NSEC */
    #
    # /* Define to 1 if `st_birthtimensec' is a member of `struct stat'. */
    # /* #undef HAVE_STRUCT_STAT_ST_BIRTHTIMENSEC */
    #
    # /* Define to 1 if `st_birthtimespec.tv_nsec' is a member of `struct stat'. */
    # #define HAVE_STRUCT_STAT_ST_BIRTHTIMESPEC_TV_NSEC 1
    HAVE_STRUCT_STAT_ST_BIRTHTIMESPEC_TV_NSEC = 1
    #
    # /* Define to 1 if `st_birthtim.tv_nsec' is a member of `struct stat'. */
    # /* #undef HAVE_STRUCT_STAT_ST_BIRTHTIM_TV_NSEC */
    #
    # /* Define to 1 if `tm_zone' is a member of `struct tm'. */
    # #define HAVE_STRUCT_TM_TM_ZONE 1
    HAVE_STRUCT_TM_TM_ZONE = 1
    #
    # /* Define to 1 if `unicode' is a member of `struct unipair'. */
    # /* #undef HAVE_STRUCT_UNIPAIR_UNICODE */
    #
    # /* Define to 1 if you have the `symlink' function. */
    # #define HAVE_SYMLINK 1
    HAVE_SYMLINK = 1
    #
    # /* Define to 1 if you have the `sync' function. */
    # #define HAVE_SYNC 1
    HAVE_SYNC = 1
    #
    # /* Define to 1 if you have the `sysctl' function. */
    # #define HAVE_SYSCTL 1
    HAVE_SYSCTL = 1
    #
    # /* Define to 1 if you have the `sysmp' function. */
    # /* #undef HAVE_SYSMP */
    #
    # /* Define to 1 if you have the <sys/acl.h> header file. */
    # #define HAVE_SYS_ACL_H 1
    HAVE_SYS_ACL_H = 1
    #
    # /* Define to 1 if you have the <sys/bitypes.h> header file. */
    # /* #undef HAVE_SYS_BITYPES_H */
    #
    # /* Define to 1 if you have the <sys/cdefs.h> header file. */
    # #define HAVE_SYS_CDEFS_H 1
    HAVE_SYS_CDEFS_H = 1
    #
    # /* Define to 1 if you have the <sys/fs/s5param.h> header file. */
    # /* #undef HAVE_SYS_FS_S5PARAM_H */
    #
    # /* Define to 1 if you have the <sys/fs_types.h> header file. */
    # /* #undef HAVE_SYS_FS_TYPES_H */
    #
    # /* Define to 1 if you have the <sys/inttypes.h> header file. */
    # /* #undef HAVE_SYS_INTTYPES_H */
    #
    # /* Define to 1 if you have the <sys/loadavg.h> header file. */
    # /* #undef HAVE_SYS_LOADAVG_H */
    #
    # /* Define to 1 if you have the <sys/mount.h> header file. */
    # #define HAVE_SYS_MOUNT_H 1
    HAVE_SYS_MOUNT_H = 1
    #
    # /* Define to 1 if you have the <sys/param.h> header file. */
    # #define HAVE_SYS_PARAM_H 1
    HAVE_SYS_PARAM_H = 1
    #
    # /* Define to 1 if you have the <sys/pstat.h> header file. */
    # /* #undef HAVE_SYS_PSTAT_H */
    #
    # /* Define to 1 if you have the <sys/random.h> header file. */
    # #define HAVE_SYS_RANDOM_H 1
    HAVE_SYS_RANDOM_H = 1
    #
    # /* Define to 1 if you have the <sys/resource.h> header file. */
    # #define HAVE_SYS_RESOURCE_H 1
    HAVE_SYS_RESOURCE_H = 1
    #
    # /* Define to 1 if you have the <sys/select.h> header file. */
    # #define HAVE_SYS_SELECT_H 1
    HAVE_SYS_SELECT_H = 1
    #
    # /* Define to 1 if you have the <sys/socket.h> header file. */
    # #define HAVE_SYS_SOCKET_H 1
    HAVE_SYS_SOCKET_H = 1
    #
    # /* Define to 1 if you have the <sys/soundcard.h> header file. */
    # /* #undef HAVE_SYS_SOUNDCARD_H */
    #
    # /* Define to 1 if you have the <sys/statfs.h> header file. */
    # /* #undef HAVE_SYS_STATFS_H */
    #
    # /* Define to 1 if you have the <sys/stat.h> header file. */
    # #define HAVE_SYS_STAT_H 1
    HAVE_SYS_STAT_H = 1
    #
    # /* Define to 1 if you have the <sys/sysctl.h> header file. */
    # #define HAVE_SYS_SYSCTL_H 1
    HAVE_SYS_SYSCTL_H = 1
    #
    # /* Define to 1 if you have the <sys/sysinfo.h> header file. */
    # /* #undef HAVE_SYS_SYSINFO_H */
    #
    # /* Define to 1 if you have the <sys/sysmp.h> header file. */
    # /* #undef HAVE_SYS_SYSMP_H */
    #
    # /* Define to 1 if you have the <sys/systeminfo.h> header file. */
    # /* #undef HAVE_SYS_SYSTEMINFO_H */
    #
    # /* Define to 1 if you have the <sys/time.h> header file. */
    # #define HAVE_SYS_TIME_H 1
    HAVE_SYS_TIME_H = 1
    #
    # /* Define to 1 if you have the <sys/types.h> header file. */
    # #define HAVE_SYS_TYPES_H 1
    HAVE_SYS_TYPES_H = 1
    #
    # /* Define to 1 if you have the <sys/un.h> header file. */
    # #define HAVE_SYS_UN_H 1
    HAVE_SYS_UN_H = 1
    #
    # /* Define to 1 if you have the <sys/utsname.h> header file. */
    # #define HAVE_SYS_UTSNAME_H 1
    HAVE_SYS_UTSNAME_H = 1
    #
    # /* Define to 1 if you have the <sys/vfs.h> header file. */
    # /* #undef HAVE_SYS_VFS_H */
    #
    # /* Define to 1 if you have the <sys/vlimit.h> header file. */
    # /* #undef HAVE_SYS_VLIMIT_H */
    #
    # /* Define to 1 if you have <sys/wait.h> that is POSIX.1 compatible. */
    # #define HAVE_SYS_WAIT_H 1
    HAVE_SYS_WAIT_H = 1
    #
    # /* Define to 1 if you have the <term.h> header file. */
    # #define HAVE_TERM_H 1
    HAVE_TERM_H = 1
    #
    # /* Define to 1 if you have the tiff library (-ltiff). */
    # /* #undef HAVE_TIFF */
    #
    # /* Define to 1 if you have the `timegm' function. */
    # #define HAVE_TIMEGM 1
    HAVE_TIMEGM = 1
    #
    # /* Define to 1 if timerfd functions are supported as in GNU/Linux. */
    # /* #undef HAVE_TIMERFD */
    #
    # /* Define to 1 if you have the `timer_getoverrun' function. */
    # /* #undef HAVE_TIMER_GETOVERRUN */
    #
    # /* Define to 1 if you have the `timer_settime' function. */
    # /* #undef HAVE_TIMER_SETTIME */
    #
    # /* Define to 1 if the system has the type `timezone_t'. */
    # /* #undef HAVE_TIMEZONE_T */
    #
    # /* Define to 1 if speed_t has some sort of nonsensically tiny size. */
    # /* #undef HAVE_TINY_SPEED_T */
    #
    # /* Define if struct tm has the tm_gmtoff member. */
    # #define HAVE_TM_GMTOFF 1
    HAVE_TM_GMTOFF = 1
    #
    # /* Define to 1 if your `struct tm' has `tm_zone'. Deprecated, use
    #    `HAVE_STRUCT_TM_TM_ZONE' instead. */
    # #define HAVE_TM_ZONE 1
    HAVE_TM_ZONE = 1
    #
    # /* Define to 1 if you have the `touchlock' function. */
    # /* #undef HAVE_TOUCHLOCK */
    #
    # /* Define to 1 if you have the `trunc' function. */
    # #define HAVE_TRUNC 1
    HAVE_TRUNC = 1
    #
    # /* Define to 1 if typeof works with your compiler. */
    # #define HAVE_TYPEOF 1
    HAVE_TYPEOF = 1
    #
    # /* Define to 1 if you don't have `tm_zone' but do have the external array
    #    `tzname'. */
    # /* #undef HAVE_TZNAME */
    #
    # /* Define if Emacs supports unexec. */
    # /* #undef HAVE_UNEXEC */
    #
    # /* Define to 1 if you have the <unistd.h> header file. */
    # #define HAVE_UNISTD_H 1
    HAVE_UNISTD_H = 1
    #
    # /* Define to 1 if the system has the type 'unsigned long long int'. */
    # #define HAVE_UNSIGNED_LONG_LONG_INT 1
    HAVE_UNSIGNED_LONG_LONG_INT = 1
    #
    # /* Define to 1 if you have the <util.h> header file. */
    # #define HAVE_UTIL_H 1
    HAVE_UTIL_H = 1
    #
    # /* Define to 1 if you have the `utimensat' function. */
    # #define HAVE_UTIMENSAT 1
    HAVE_UTIMENSAT = 1
    #
    # /* Define to 1 if you have the <utmp.h> header file. */
    # #define HAVE_UTMP_H 1
    HAVE_UTMP_H = 1
    #
    # /* Define to 1 if you have the <valgrind/valgrind.h> header file. */
    # /* #undef HAVE_VALGRIND_VALGRIND_H */
    #
    # /* Define to 1 if you have the `vfork' function. */
    # #define HAVE_VFORK 1
    HAVE_VFORK = 1
    #
    # /* Define to 1 if you have the <vfork.h> header file. */
    # /* #undef HAVE_VFORK_H */
    #
    # /* Define to 1 to use w32notify. */
    # /* #undef HAVE_W32NOTIFY */
    #
    # /* Define to 1 if you have the <wchar.h> header file. */
    # #define HAVE_WCHAR_H 1
    HAVE_WCHAR_H = 1
    #
    # /* Define if you have the 'wchar_t' type. */
    # #define HAVE_WCHAR_T 1
    HAVE_WCHAR_T = 1
    #
    # /* Define to 1 if using libwebp. */
    # #define HAVE_WEBP 1
    HAVE_WEBP = 1
    #
    # /* Define if you have a window system. */
    # #define HAVE_WINDOW_SYSTEM 1
    HAVE_WINDOW_SYSTEM = 1
    #
    # /* Define to 1 if you have the <winsock2.h> header file. */
    # /* #undef HAVE_WINSOCK2_H */
    #
    # /* Define to 1 if `fork' works. */
    # #define HAVE_WORKING_FORK 1
    HAVE_WORKING_FORK = 1
    #
    # /* Define to 1 if fstatat (..., 0) works. For example, it does not work in AIX
    #    7.1. */
    # #define HAVE_WORKING_FSTATAT_ZERO_FLAG 1
    HAVE_WORKING_FSTATAT_ZERO_FLAG = 1
    #
    # /* Define if utimes works properly. */
    # #define HAVE_WORKING_UTIMES 1
    HAVE_WORKING_UTIMES = 1
    #
    # /* Define to 1 if `vfork' works. */
    # #define HAVE_WORKING_VFORK 1
    HAVE_WORKING_VFORK = 1
    #
    # /* Define to 1 if you have the <ws2tcpip.h> header file. */
    # /* #undef HAVE_WS2TCPIP_H */
    #
    # /* Define to 1 if you want to use version 11 of X windows. */
    # /* #undef HAVE_X11 */
    #
    # /* Define to 1 if you have the X11R6 or newer version of Xlib. */
    # /* #undef HAVE_X11R6 */
    #
    # /* Define if you have usable X11R6-style XIM support. */
    # /* #undef HAVE_X11R6_XIM */
    #
    # /* Define to 1 if you have the X11R6 or newer version of Xt. */
    # /* #undef HAVE_X11XTR6 */
    #
    # /* Define to 1 if you have the Xaw3d library (-lXaw3d). */
    # /* #undef HAVE_XAW3D */
    #
    # /* Define to 1 if you have the Xdbe extension. */
    # /* #undef HAVE_XDBE */
    #
    # /* Define to 1 if you have the Xfixes extension. */
    # /* #undef HAVE_XFIXES */
    #
    # /* Define to 1 if you have the Xft library. */
    # /* #undef HAVE_XFT */
    #
    # /* Define to 1 if XIM is available */
    # /* #undef HAVE_XIM */
    #
    # /* Define to 1 if you have the Xinerama extension. */
    # /* #undef HAVE_XINERAMA */
    #
    # /* Define to 1 if the X Input Extension version 2.0 or later is present. */
    # /* #undef HAVE_XINPUT2 */
    #
    # /* Define to 1 if you have the Xkb extension. */
    # /* #undef HAVE_XKB */
    #
    # /* Define to 1 if you have the Xpm library (-lXpm). */
    # /* #undef HAVE_XPM */
    #
    # /* Define to 1 if you have the XRandr extension. */
    # /* #undef HAVE_XRANDR */
    #
    # /* Define to 1 if XRender is available. */
    # /* #undef HAVE_XRENDER */
    #
    # /* Define to 1 if you have the `XrmSetDatabase' function. */
    # /* #undef HAVE_XRMSETDATABASE */
    #
    # /* Define to 1 if you have the `XScreenNumberOfScreen' function. */
    # /* #undef HAVE_XSCREENNUMBEROFSCREEN */
    #
    # /* Define to 1 if you have the `XScreenResourceString' function. */
    # /* #undef HAVE_XSCREENRESOURCESTRING */
    #
    # /* Define to 1 if you have xwidgets support. */
    # /* #undef HAVE_XWIDGETS */
    #
    # /* Define if you have usable i18n support. */
    # /* #undef HAVE_X_I18N */
    #
    # /* Define to 1 if you have the SM library (-lSM). */
    # /* #undef HAVE_X_SM */
    #
    # /* Define to 1 if you want to use the X window system. */
    # /* #undef HAVE_X_WINDOWS */
    #
    # /* Define to 1 if you have the zlib library (-lz). */
    # #define HAVE_ZLIB 1
    HAVE_ZLIB = 1
    #
    # /* Define to 1 if _setjmp and _longjmp work. */
    # #define HAVE__SETJMP 1
    HAVE__SETJMP = 1
    #
    # /* Define to 1 if the compiler supports __builtin_expect,
    #    and to 2 if <builtins.h> does.  */
    # #define HAVE___BUILTIN_EXPECT 1
    HAVE___BUILTIN_EXPECT = 1
    # #ifndef HAVE___BUILTIN_EXPECT
    # # define __builtin_expect(e, c) (e)
    # #elif HAVE___BUILTIN_EXPECT == 2
    # # include <builtins.h>
    # #endif
    #
    #
    # /* Define to 1 if you have the '__builtin_frame_address' function. */
    # #define HAVE___BUILTIN_FRAME_ADDRESS 1
    HAVE___BUILTIN_FRAME_ADDRESS = 1
    #
    # /* Define to 1 if you have the '__builtin_unwind_init' function. */
    # #define HAVE___BUILTIN_UNWIND_INIT 1
    HAVE___BUILTIN_UNWIND_INIT = 1
    #
    # /* Define to 1 if you have the `__executable_start' function. */
    # /* #undef HAVE___EXECUTABLE_START */
    #
    # /* Define to 1 if the compiler supports the keyword '__inline'. */
    # #define HAVE___INLINE 1
    HAVE___INLINE = 1
    #
    # /* Define to 1 if you have the `__lsan_ignore_object' function. */
    # /* #undef HAVE___LSAN_IGNORE_OBJECT */
    #
    # /* Define to support using a Hesiod database to find the POP server. */
    # /* #undef HESIOD */
    #
    # /* Define if the system is HPUX. */
    # /* #undef HPUX */
    #
    # /* Define to use gmalloc before dumping and the system malloc after. */
    # /* #undef HYBRID_MALLOC */
    #
    # /* This is substituted when $TERM is "internal". */
    # /* #undef INTERNAL_TERMINAL */
    #
    # /* Define to read input using SIGIO. */
    # #define INTERRUPT_INPUT 1
    INTERRUPT_INPUT = 1
    #
    # /* Returns true if character is any form of separator. */
    # #define IS_ANY_SEP(_c_) (IS_DIRECTORY_SEP (_c_))
    @staticmethod
    def IS_ANY_SEP(c):
        return PP.IS_DIRECTORY_SEP(c)

    # /* Returns true if character is a device separator. */
    # #define IS_DEVICE_SEP(_c_) 0
    @staticmethod
    def IS_DEVICE_SEP(c):
        return 0

    # /* Returns true if character is a directory separator. */
    # #define IS_DIRECTORY_SEP(_c_) ((_c_) == DIRECTORY_SEP)
    @staticmethod
    def IS_DIRECTORY_SEP(c):
        return c == PP.DIRECTORY_SEP


    # /* Define to support Kerberos-authenticated POP mail retrieval. */
    # /* #undef KERBEROS */
    #
    # /* Define to use Kerberos 5 instead of Kerberos 4. */
    # /* #undef KERBEROS5 */
    #
    # /* Define to 1 if Linux sysinfo sizes are in multiples of mem_unit bytes. */
    # /* #undef LINUX_SYSINFO_UNIT */
    #
    # /* Define to 1 if 'lstat' dereferences a symlink specified with a trailing
    #    slash. */
    # /* #undef LSTAT_FOLLOWS_SLASHED_SYMLINK */
    #
    # /* String giving fallback POP mail host. */
    # /* #undef MAILHOST */
    #
    # /* Define to unlink, rather than empty, mail spool after reading. */
    # /* #undef MAIL_UNLINK_SPOOL */
    #
    # /* Define if the mailer uses flock to interlock the mail spool. */
    # #define MAIL_USE_FLOCK 1
    MAIL_USE_FLOCK = 1
    #
    # /* Define if the mailer uses lockf to interlock the mail spool. */
    # /* #undef MAIL_USE_LOCKF */
    #
    # /* Define to support POP mail retrieval. */
    # /* #undef MAIL_USE_POP */
    #
    # /* If malloc(0) is != NULL, define this to 1. Otherwise define this to 0. */
    # #define MALLOC_0_IS_NONNULL 1
    MALLOC_0_IS_NONNULL = 1
    #
    # /* Use GNU style printf and scanf.  */
    # #ifndef __USE_MINGW_ANSI_STDIO
    # # define __USE_MINGW_ANSI_STDIO 1
    # #endif
    #
    #
    # /* Alternative system extension for dynamic libraries. */
    # #define MODULES_SECONDARY_SUFFIX ".so"
    MODULES_SECONDARY_SUFFIX = ".so"
    #
    # /* System extension for dynamic libraries */
    # #define MODULES_SUFFIX ".dylib"
    MODULES_SUFFIX = ".dylib"
    #
    # /* Define if the system is MS DOS. */
    # /* #undef MSDOS */
    #
    # /* Define if system's imake configuration file defines 'NeedWidePrototypes' as
    #    'NO'. */
    # /* #undef NARROWPROTO */
    #
    # /* System extension for native compiled elisp */
    # #define NATIVE_ELISP_SUFFIX ".eln"
    NATIVE_ELISP_SUFFIX = ".eln"
    #
    # /* Define if ObjC compiler supports instancetype natively. */
    # #define NATIVE_OBJC_INSTANCETYPE 1
    NATIVE_OBJC_INSTANCETYPE = 1
    #
    # /* Define to 1 if fchmodat+AT_SYMLINK_NOFOLLOW does not work right on
    #    non-symlinks. */
    # /* #undef NEED_FCHMODAT_NONSYMLINK_FIX */
    #
    # /* Define if the compilation of mktime.c should define 'mktime_internal'. */
    # #define NEED_MKTIME_INTERNAL 1
    NEED_MKTIME_INTERNAL = 1
    #
    # /* Define if the compilation of mktime.c should define 'mktime' with the
    #    native Windows TZ workaround. */
    # /* #undef NEED_MKTIME_WINDOWS */
    #
    # /* Define if the compilation of mktime.c should define 'mktime' with the
    #    algorithmic workarounds. */
    # #define NEED_MKTIME_WORKING 1
    NEED_MKTIME_WORKING = 1
    #
    # /* Define to 1 if your C compiler doesn't accept -c and -o together. */
    # /* #undef NO_MINUS_C_MINUS_O */
    #
    # /* Minimum value of NSIG. */
    # /* #undef NSIG_MINIMUM */
    #
    # /* Define to 1 if you are using NS windowing under macOS. */
    # #define NS_IMPL_COCOA 1
    NS_IMPL_COCOA = 1
    #
    # /* Define to 1 if you are using NS windowing under GNUstep. */
    # /* #undef NS_IMPL_GNUSTEP */
    #
    # /* Build an NS bundled app */
    # #define NS_SELF_CONTAINED 1
    NS_SELF_CONTAINED = 1
    #
    # /* Name of the file to open to get a null file, or a data sink. */
    # #define NULL_DEVICE "/dev/null"
    NULL_DEVICE = "/dev/null"
    #
    # /* Define to 1 if the nlist n_name member is a pointer */
    # /* #undef N_NAME_POINTER */
    #
    # /* Define to 1 if open() fails to recognize a trailing slash. */
    # #define OPEN_TRAILING_SLASH_BUG 1
    OPEN_TRAILING_SLASH_BUG = 1
    #
    # /* Define to the address where bug reports for this package should be sent. */
    # #define PACKAGE_BUGREPORT "bug-gnu-emacs@gnu.org"
    PACKAGE_BUGREPORT = "bug-gnu-emacs@gnu.org"
    #
    # /* Define to the full name of this package. */
    # #define PACKAGE_NAME "GNU Emacs"
    PACKAGE_NAME = "GNU Emacs"
    #
    # /* Define to the full name and version of this package. */
    # #define PACKAGE_STRING "GNU Emacs 29.0.50"
    PACKAGE_STRING = "GNU Emacs 29.0.50"
    #
    # /* Define to the one symbol short name of this package. */
    # #define PACKAGE_TARNAME "emacs"
    PACKAGE_TARNAME = "emacs"
    #
    # /* Define to the home page for this package. */
    # #define PACKAGE_URL "https://www.gnu.org/software/emacs/"
    PACKAGE_URL = "https://www.gnu.org/software/emacs/"
    #
    # /* Define to the version of this package. */
    # #define PACKAGE_VERSION "29.0.50"
    PACKAGE_VERSION = "29.0.50"
    #
    # /* Define to empty to suppress deprecation warnings when building with
    #    --enable-gcc-warnings and with libpng versions before 1.5, which lack
    #    png_longjmp. */
    # /* #undef PNG_DEPSTRUCT */
    #
    # /* Define if you poll periodically to detect C-g. */
    # #define POLL_FOR_INPUT 1
    POLL_FOR_INPUT = 1
    #
    # /* Define to the type that is the result of default argument promotions of
    #    type mode_t. */
    # #define PROMOTED_MODE_T int
    PROMOTED_MODE_T = int
    #
    # /* Define to 1 if pthread_sigmask(), when it fails, returns -1 and sets errno.
    #    */
    # /* #undef PTHREAD_SIGMASK_FAILS_WITH_ERRNO */
    #
    # /* Define to 1 if pthread_sigmask may return 0 and have no effect. */
    # /* #undef PTHREAD_SIGMASK_INEFFECTIVE */
    #
    # /* Define to 1 if pthread_sigmask() unblocks signals incorrectly. */
    # /* #undef PTHREAD_SIGMASK_UNBLOCK_BUG */
    #
    # /* Define to l, ll, u, ul, ull, etc., as suitable for constants of type
    #    'ptrdiff_t'. */
    # /* #undef PTRDIFF_T_SUFFIX */
    #
    # /* How to iterate over PTYs. */
    # #define PTY_ITERATION int i; for (i = 0; i < 1; i++)
    PTY_ITERATION = "int i; for (i = 0; i < 1; i++)"
    #
    # /* How to get the device name of the control end of a PTY, if non-standard. */
    # #define PTY_NAME_SPRINTF /**/
    PTY_NAME_SPRINTF = None
    #
    # /* How to open a PTY, if non-standard. */
    # #define PTY_OPEN do { fd = posix_openpt (O_RDWR | O_CLOEXEC | O_NOCTTY); if (fd < 0 && errno == EINVAL) fd = posix_openpt (O_RDWR | O_NOCTTY); } while (false)
    PTY_OPEN = "do { fd = posix_openpt (O_RDWR | O_CLOEXEC | O_NOCTTY); if (fd < 0 && errno == EINVAL) fd = posix_openpt (O_RDWR | O_NOCTTY); } while (false)"
    #
    # /* How to get device name of the tty end of a PTY, if non-standard. */
    # #define PTY_TTY_NAME_SPRINTF { char *ptyname = 0; sigset_t blocked; sigemptyset (&blocked); sigaddset (&blocked, SIGCHLD); pthread_sigmask (SIG_BLOCK, &blocked, 0); if (grantpt (fd) != -1 && unlockpt (fd) != -1) ptyname = ptsname(fd); pthread_sigmask (SIG_UNBLOCK, &blocked, 0); if (!ptyname) { emacs_close (fd); return -1; } snprintf (pty_name, PTY_NAME_SIZE, "%s", ptyname); }
    PTY_TTY_NAME_SPRINTF = """{ char *ptyname = 0; sigset_t blocked; sigemptyset (&blocked); sigaddset (&blocked, SIGCHLD); pthread_sigmask (SIG_BLOCK, &blocked, 0); if (grantpt (fd) != -1 && unlockpt (fd) != -1) ptyname = ptsname(fd); pthread_sigmask (SIG_UNBLOCK, &blocked, 0); if (!ptyname) { emacs_close (fd); return -1; } snprintf (pty_name, PTY_NAME_SIZE, "%s", ptyname); }"""
    #
    # /* Define to 1 if readlink fails to recognize a trailing slash. */
    # #define READLINK_TRAILING_SLASH_BUG 1
    READLINK_TRAILING_SLASH_BUG = 1
    #
    # /* Define to 1 if readlink sets errno instead of truncating a too-long link.
    #    */
    # /* #undef READLINK_TRUNCATE_BUG */
    #
    # /* Define REL_ALLOC if you want to use the relocating allocator for buffer
    #    space. */
    # /* #undef REL_ALLOC */
    #
    # /* Define to 1 if gnulib's dirfd() replacement is used. */
    # /* #undef REPLACE_DIRFD */
    #
    # /* Define if emacs.c needs to call run_time_remap; for HPUX. */
    # /* #undef RUN_TIME_REMAP */
    #
    # /* Character that separates PATH elements. */
    # #define SEPCHAR ':'
    SEPCHAR = _os.sep
    #
    # /* How to set up a slave PTY, if needed. */
    # /* #undef SETUP_SLAVE_PTY */
    #
    # /* Make process_send_signal work by "typing" a signal character on the pty. */
    # #define SIGNALS_VIA_CHARACTERS 1
    SIGNALS_VIA_CHARACTERS = 1
    #
    # /* Define to l, ll, u, ul, ull, etc., as suitable for constants of type
    #    'sig_atomic_t'. */
    # /* #undef SIG_ATOMIC_T_SUFFIX */
    #
    # /* The size of `speed_t', as computed by sizeof. */
    # #define SIZEOF_SPEED_T 8
    SIZEOF_SPEED_T = 8
    #
    # /* Define to l, ll, u, ul, ull, etc., as suitable for constants of type
    #    'size_t'. */
    # /* #undef SIZE_T_SUFFIX */
    #
    # /* Define if the system is Solaris. */
    # /* #undef SOLARIS2 */
    #
    # /* If using the C implementation of alloca, define if you know the
    #    direction of stack growth for your system; otherwise it will be
    #    automatically deduced at runtime.
    # 	STACK_DIRECTION > 0 => grows toward higher addresses
    # 	STACK_DIRECTION < 0 => grows toward lower addresses
    # 	STACK_DIRECTION = 0 => direction of growth unknown */
    # /* #undef STACK_DIRECTION */
    #
    # /* Define if the block counts reported by statfs may be truncated to 2GB and
    #    the correct values may be stored in the f_spare array. (SunOS 4.1.2, 4.1.3,
    #    and 4.1.3_U1 are reported to have this problem. SunOS 4.1.1 seems not to be
    #    affected.) */
    # /* #undef STATFS_TRUNCATES_BLOCK_COUNTS */
    #
    # /* Define to 1 if the `S_IS*' macros in <sys/stat.h> do not work properly. */
    # /* #undef STAT_MACROS_BROKEN */
    #
    # /* Define if statfs takes 2 args and struct statfs has a field named f_bsize.
    #    (4.3BSD, SunOS 4, HP-UX) */
    # #define STAT_STATFS2_BSIZE 1
    STAT_STATFS2_BSIZE = 1
    #
    # /* Define if statfs takes 2 args and struct statfs has a field named f_frsize.
    #    (glibc/Linux > 2.6) */
    # /* #undef STAT_STATFS2_FRSIZE */
    #
    # /* Define if statfs takes 2 args and struct statfs has a field named f_fsize.
    #    (4.4BSD, NetBSD) */
    # /* #undef STAT_STATFS2_FSIZE */
    #
    # /* Define if statfs takes 3 args. (DEC Alpha running OSF/1) */
    # /* #undef STAT_STATFS3_OSF1 */
    #
    # /* Define if statfs takes 4 args. (SVR3, old Irix) */
    # /* #undef STAT_STATFS4 */
    #
    # /* Define if there is a function named statvfs. (SVR4) */
    # /* #undef STAT_STATVFS */
    #
    # /* Define if statvfs64 should be preferred over statvfs. */
    # /* #undef STAT_STATVFS64 */
    #
    # /* Define to 1 if all of the C90 standard headers exist (not just the ones
    #    required in a freestanding environment). This macro is provided for
    #    backward compatibility; new code need not use it. */
    # #define STDC_HEADERS 1
    STDC_HEADERS = 1
    #
    # /* Define to 1 on System V Release 4. */
    # /* #undef SVR4 */
    #
    # /* Define to 1 to use the system memory allocator, even if it is not Doug Lea
    #    style. */
    # #define SYSTEM_MALLOC 1
    SYSTEM_MALLOC = 1
    #
    # /* The type of system you are compiling for; sets 'system-type'. */
    # #define SYSTEM_TYPE "darwin"
    SYSTEM_TYPE = "darwin"
    #
    # /* Undocumented. */
    # #define TAB3 OXTABS
    TAB3 = "OXTABS"
    #
    # /* Undocumented. */
    # /* #undef TABDLY */
    #
    # /* Define to 1 if you use terminfo instead of termcap. */
    # #define TERMINFO 1
    TERMINFO = 1
    #
    # /* Define to 1 if the terminfo library defines the variables BC, PC, and UP.
    #    */
    # #define TERMINFO_DEFINES_BC 1
    TERMINFO_DEFINES_BC = 1
    #
    # /* Define to the header for the built-in window system. */
    # #define TERM_HEADER "nsterm.h"
    TERM_HEADER = "nsterm.h"
    #
    # /* Define to 1 if you want elisp thread support. */
    # #define THREADS_ENABLED 1
    THREADS_ENABLED = 1
    #
    # /* Define to 1 if time_t is signed. */
    # #define TIME_T_IS_SIGNED 1
    TIME_T_IS_SIGNED = 1
    #
    # /* Some platforms redefine this. */
    # /* #undef TIOCSIGSEND */
    #
    # /* Define to 1 if your <sys/time.h> declares `struct tm'. */
    # /* #undef TM_IN_SYS_TIME */
    #
    # /* Define to 1 if the type of the st_atim member of a struct stat is struct
    #    timespec. */
    # /* #undef TYPEOF_STRUCT_STAT_ST_ATIM_IS_STRUCT_TIMESPEC */
    #
    # /* Define to 1 for Encore UMAX. */
    # /* #undef UMAX */
    #
    # /* Define to 1 for Encore UMAX 4.3 that has <inq_status/cpustats.h> instead of
    #    <sys/cpustats.h>. */
    # /* #undef UMAX4_3 */
    #
    # /* Define if the system has Unix98 PTYs. */
    # #define UNIX98_PTYS 1
    UNIX98_PTYS = 1
    #
    # /* Define to 1 if FIONREAD is usable. */
    # #define USABLE_FIONREAD 1
    USABLE_FIONREAD = 1
    #
    # /* Define to 1 if SIGIO is usable. */
    # #define USABLE_SIGIO 1
    USABLE_SIGIO = 1
    #
    # /* Define to 1 if SIGPOLL is usable but SIGIO is not. */
    # /* #undef USABLE_SIGPOLL */
    #
    # /* How to get a user's full name. */
    # #define USER_FULL_NAME pw->pw_gecos
    USER_FULL_NAME = "pw->pw_gecos"
    #
    # /* Define to nonzero if you want access control list support. */
    # #define USE_ACL 1
    USE_ACL = 1
    #
    # /* Define to 1 if using cairo on Haiku. */
    # /* #undef USE_BE_CAIRO */
    #
    # /* Define to 1 if using cairo. */
    # /* #undef USE_CAIRO */
    #
    # /* Define to 1 if using file notifications. */
    # #define USE_FILE_NOTIFY 1
    USE_FILE_NOTIFY = 1
    #
    # /* Define to 1 if using GTK. */
    # /* #undef USE_GTK */
    #
    # /* Define to 1 if using the Lucid X toolkit. */
    # /* #undef USE_LUCID */
    #
    # /* Define to use mmap to allocate buffer text. */
    # /* #undef USE_MMAP_FOR_BUFFERS */
    #
    # /* Define to 1 if using the Motif X toolkit. */
    # /* #undef USE_MOTIF */
    #
    # /* Define to 1 if you use ncurses. */
    # #define USE_NCURSES 1
    USE_NCURSES = 1
    #
    # /* Enable extensions on AIX 3, Interix.  */
    # #ifndef _ALL_SOURCE
    # # define _ALL_SOURCE 1
    # #endif
    # /* Enable general extensions on macOS.  */
    # #ifndef _DARWIN_C_SOURCE
    # # define _DARWIN_C_SOURCE 1
    # #endif
    # /* Enable general extensions on Solaris.  */
    # #ifndef __EXTENSIONS__
    # # define __EXTENSIONS__ 1
    # #endif
    # /* Enable GNU extensions on systems that have them.  */
    # #ifndef _GNU_SOURCE
    # # define _GNU_SOURCE 1
    # #endif
    # /* Enable X/Open compliant socket functions that do not require linking
    #    with -lxnet on HP-UX 11.11.  */
    # #ifndef _HPUX_ALT_XOPEN_SOCKET_API
    # # define _HPUX_ALT_XOPEN_SOCKET_API 1
    # #endif
    # /* Identify the host operating system as Minix.
    #    This macro does not affect the system headers' behavior.
    #    A future release of Autoconf may stop defining this macro.  */
    # #ifndef _MINIX
    # /* # undef _MINIX */
    # #endif
    # /* Enable general extensions on NetBSD.
    #    Enable NetBSD compatibility extensions on Minix.  */
    # #ifndef _NETBSD_SOURCE
    # # define _NETBSD_SOURCE 1
    # #endif
    # /* Enable OpenBSD compatibility extensions on NetBSD.
    #    Oddly enough, this does nothing on OpenBSD.  */
    # #ifndef _OPENBSD_SOURCE
    # # define _OPENBSD_SOURCE 1
    # #endif
    # /* Define to 1 if needed for POSIX-compatible behavior.  */
    # #ifndef _POSIX_SOURCE
    # /* # undef _POSIX_SOURCE */
    # #endif
    # /* Define to 2 if needed for POSIX-compatible behavior.  */
    # #ifndef _POSIX_1_SOURCE
    # /* # undef _POSIX_1_SOURCE */
    # #endif
    # /* Enable POSIX-compatible threading on Solaris.  */
    # #ifndef _POSIX_PTHREAD_SEMANTICS
    # # define _POSIX_PTHREAD_SEMANTICS 1
    # #endif
    # /* Enable extensions specified by ISO/IEC TS 18661-5:2014.  */
    # #ifndef __STDC_WANT_IEC_60559_ATTRIBS_EXT__
    # # define __STDC_WANT_IEC_60559_ATTRIBS_EXT__ 1
    # #endif
    # /* Enable extensions specified by ISO/IEC TS 18661-1:2014.  */
    # #ifndef __STDC_WANT_IEC_60559_BFP_EXT__
    # # define __STDC_WANT_IEC_60559_BFP_EXT__ 1
    # #endif
    # /* Enable extensions specified by ISO/IEC TS 18661-2:2015.  */
    # #ifndef __STDC_WANT_IEC_60559_DFP_EXT__
    # # define __STDC_WANT_IEC_60559_DFP_EXT__ 1
    # #endif
    # /* Enable extensions specified by ISO/IEC TS 18661-4:2015.  */
    # #ifndef __STDC_WANT_IEC_60559_FUNCS_EXT__
    # # define __STDC_WANT_IEC_60559_FUNCS_EXT__ 1
    # #endif
    # /* Enable extensions specified by ISO/IEC TS 18661-3:2015.  */
    # #ifndef __STDC_WANT_IEC_60559_TYPES_EXT__
    # # define __STDC_WANT_IEC_60559_TYPES_EXT__ 1
    # #endif
    # /* Enable extensions specified by ISO/IEC TR 24731-2:2010.  */
    # #ifndef __STDC_WANT_LIB_EXT2__
    # # define __STDC_WANT_LIB_EXT2__ 1
    # #endif
    # /* Enable extensions specified by ISO/IEC 24747:2009.  */
    # #ifndef __STDC_WANT_MATH_SPEC_FUNCS__
    # # define __STDC_WANT_MATH_SPEC_FUNCS__ 1
    # #endif
    # /* Enable extensions on HP NonStop.  */
    # #ifndef _TANDEM_SOURCE
    # # define _TANDEM_SOURCE 1
    # #endif
    # /* Enable X/Open extensions.  Define to 500 only if necessary
    #    to make mbstate_t available.  */
    # #ifndef _XOPEN_SOURCE
    # /* # undef _XOPEN_SOURCE */
    # #endif
    #
    #
    # /* Define to 1 if we should use toolkit scroll bars. */
    # #define USE_TOOLKIT_SCROLL_BARS 1
    USE_TOOLKIT_SCROLL_BARS = 1
    #
    # /* An alias of GNULIB_STDIO_SINGLE_THREAD. */
    # #define USE_UNLOCKED_IO GNULIB_STDIO_SINGLE_THREAD
    USE_UNLOCKED_IO = "GNULIB_STDIO_SINGLE_THREAD"
    #
    # /* Define to 1 if you have the XCB library and X11-XCB library for mixed
    #    X11/XCB programming. */
    # /* #undef USE_XCB */
    #
    # /* Define to 1 to default runtime use of XIM to on. */
    # #define USE_XIM 1
    USE_XIM = 1
    #
    # /* Define to 1 if using an X toolkit. */
    # /* #undef USE_X_TOOLKIT */
    #
    # /* Define if the system is compatible with System III. */
    # /* #undef USG */
    #
    # /* Define if the system is compatible with System V Release 4. */
    # /* #undef USG5_4 */
    #
    # /* Define for USG systems where it works to open a pty's tty in the parent
    #    process, then close and reopen it in the child. */
    # /* #undef USG_SUBTTY_WORKS */
    #
    # /* Define to l, ll, u, ul, ull, etc., as suitable for constants of type
    #    'wchar_t'. */
    # /* #undef WCHAR_T_SUFFIX */
    #
    # /* Use long long for EMACS_INT if available. */
    # /* #undef WIDE_EMACS_INT */
    #
    # /* Define if compiling for native MS Windows. */
    # /* #undef WINDOWSNT */
    #
    # /* Define to l, ll, u, ul, ull, etc., as suitable for constants of type
    #    'wint_t'. */
    # /* #undef WINT_T_SUFFIX */
    #
    # /* Define WORDS_BIGENDIAN to 1 if your processor stores words with the most
    #    significant byte first (like Motorola and SPARC, unlike Intel). */
    # #if defined AC_APPLE_UNIVERSAL_BUILD
    # # if defined __BIG_ENDIAN__
    # #  define WORDS_BIGENDIAN 1
    # # endif
    # #else
    # # ifndef WORDS_BIGENDIAN
    # /* #  undef WORDS_BIGENDIAN */
    # # endif
    # #endif
    #
    # /* Compensate for a bug in Xos.h on some systems, where it requires time.h. */
    # /* #undef XOS_NEEDS_TIME_H */
    #
    # /* Define to the type of the 6th arg of XRegisterIMInstantiateCallback, either
    #    XPointer or XPointer*. */
    # /* #undef XRegisterIMInstantiateCallback_arg6 */
    #
    # /* Define to 1 if we should use XEditRes. */
    # /* #undef X_TOOLKIT_EDITRES */
    #
    # /* Define if the system is AIX. */
    # /* #undef _AIX */
    #
    # /* Number of bits in a file offset, on hosts where this is settable. */
    # /* #undef _FILE_OFFSET_BITS */
    #
    # /* True if the compiler says it groks GNU C version MAJOR.MINOR.  */
    # #if defined __GNUC__ && defined __GNUC_MINOR__
    # # define _GL_GNUC_PREREQ(major, minor) \
    #     ((major) < __GNUC__ + ((minor) <= __GNUC_MINOR__))
    # #else
    # # define _GL_GNUC_PREREQ(major, minor) 0
    # #endif
    #
    #
    # /* Define to 1 if <ieee754.h> is missing. */
    # #define _GL_REPLACE_IEEE754_H 1
    _GL_REPLACE_IEEE754_H = 1
    #
    # /* Define to enable the declarations of ISO C 11 types and functions. */
    # /* #undef _ISOC11_SOURCE */
    #
    # /* Define to 1 to make fseeko visible on some hosts (e.g. glibc 2.2). */
    # /* #undef _LARGEFILE_SOURCE */
    #
    # /* Define for large files, on AIX-style hosts. */
    # /* #undef _LARGE_FILES */
    #
    # /* Define if GNUstep uses ObjC exceptions. */
    # /* #undef _NATIVE_OBJC_EXCEPTIONS */
    #
    # /* The _Noreturn keyword of C11.  */
    # #ifndef _Noreturn
    # # if (defined __cplusplus \
    #       && ((201103 <= __cplusplus && !(__GNUC__ == 4 && __GNUC_MINOR__ == 7)) \
    #           || (defined _MSC_VER && 1900 <= _MSC_VER)) \
    #       && 0)
    #     /* [[noreturn]] is not practically usable, because with it the syntax
    #          extern _Noreturn void func (...);
    #        would not be valid; such a declaration would only be valid with 'extern'
    #        and '_Noreturn' swapped, or without the 'extern' keyword.  However, some
    #        AIX system header files and several gnulib header files use precisely
    #        this syntax with 'extern'.  */
    # #  define _Noreturn [[noreturn]]
    # # elif ((!defined __cplusplus || defined __clang__) \
    #         && (201112 <= (defined __STDC_VERSION__ ? __STDC_VERSION__ : 0) \
    #             || (!defined __STRICT_ANSI__ \
    #                 && (_GL_GNUC_PREREQ (4, 7) \
    #                     || (defined __apple_build_version__ \
    #                         ? 6000000 <= __apple_build_version__ \
    #                         : 3 < __clang_major__ + (5 <= __clang_minor__))))))
    #    /* _Noreturn works as-is.  */
    # # elif _GL_GNUC_PREREQ (2, 8) || defined __clang__ || 0x5110 <= __SUNPRO_C
    # #  define _Noreturn __attribute__ ((__noreturn__))
    # # elif 1200 <= (defined _MSC_VER ? _MSC_VER : 0)
    # #  define _Noreturn __declspec (noreturn)
    # # else
    # #  define _Noreturn
    # # endif
    # #endif
    #
    #
    # /* Define to 1 if your system requires this in multithreaded code. */
    # /* #undef _REENTRANT */
    #
    # /* Define if you want <regex.h> to include <limits.h>, so that it consistently
    #    overrides <limits.h>'s RE_DUP_MAX. */
    # #define _REGEX_INCLUDE_LIMITS_H 1
    _REGEX_INCLUDE_LIMITS_H = 1
    #
    # /* Define if you want regoff_t to be at least as wide POSIX requires. */
    # #define _REGEX_LARGE_OFFSETS 1
    _REGEX_LARGE_OFFSETS = 1
    #
    # /* Needed for system_process_attributes on Solaris. */
    # /* #undef _STRUCTURED_PROC */
    #
    # /* Define to 1 if your system requires this in multithreaded code. */
    # /* #undef _THREAD_SAFE */
    #
    # /* Number of bits in a timestamp, on hosts where this is settable. */
    # /* #undef _TIME_BITS */
    #
    # /* For standard stat data types on VMS. */
    # #define _USE_STD_STAT 1
    _USE_STD_STAT = 1
    #
    # /* Define to rpl_ if the getopt replacement functions and variables should be
    #    used. */
    # #define __GETOPT_PREFIX rpl_
    __GETOPT_PREFIX = "rpl_"
    #
    # /* For 64-bit time_t on 32-bit mingw. */
    # /* #undef __MINGW_USE_VC2005_COMPAT */
    #
    # /* Define to 1 if the system <stdint.h> predates C++11. */
    # /* #undef __STDC_CONSTANT_MACROS */
    #
    # /* Define to 1 if the system <stdint.h> predates C++11. */
    # /* #undef __STDC_LIMIT_MACROS */
    #
    # /* Define to 1 if C does not support variable-length arrays, and if the
    #    compiler does not already define this. */
    # /* #undef __STDC_NO_VLA__ */
    #
    # /* The _GL_ASYNC_SAFE marker should be attached to functions that are
    #    signal handlers (for signals other than SIGABRT, SIGPIPE) or can be
    #    invoked from such signal handlers.  Such functions have some restrictions:
    #      * All functions that it calls should be marked _GL_ASYNC_SAFE as well,
    #        or should be listed as async-signal-safe in POSIX
    #        <https://pubs.opengroup.org/onlinepubs/9699919799/functions/V2_chap02.html#tag_15_04>
    #        section 2.4.3.  Note that malloc(), sprintf(), and fwrite(), in
    #        particular, are NOT async-signal-safe.
    #      * All memory locations (variables and struct fields) that these functions
    #        access must be marked 'volatile'.  This holds for both read and write
    #        accesses.  Otherwise the compiler might optimize away stores to and
    #        reads from such locations that occur in the program, depending on its
    #        data flow analysis.  For example, when the program contains a loop
    #        that is intended to inspect a variable set from within a signal handler
    #            while (!signal_occurred)
    #              ;
    #        the compiler is allowed to transform this into an endless loop if the
    #        variable 'signal_occurred' is not declared 'volatile'.
    #    Additionally, recall that:
    #      * A signal handler should not modify errno (except if it is a handler
    #        for a fatal signal and ends by raising the same signal again, thus
    #        provoking the termination of the process).  If it invokes a function
    #        that may clobber errno, it needs to save and restore the value of
    #        errno.  */
    # #define _GL_ASYNC_SAFE
    _GL_ASYNC_SAFE = None
    #
    #
    # /* Attributes.  */
    # #if (defined __has_attribute \
    #      && (!defined __clang_minor__ \
    #          || 3 < __clang_major__ + (5 <= __clang_minor__)))
    # # define _GL_HAS_ATTRIBUTE(attr) __has_attribute (__##attr##__)
    # #else
    # # define _GL_HAS_ATTRIBUTE(attr) _GL_ATTR_##attr
    # # define _GL_ATTR_alloc_size _GL_GNUC_PREREQ (4, 3)
    # # define _GL_ATTR_always_inline _GL_GNUC_PREREQ (3, 2)
    # # define _GL_ATTR_artificial _GL_GNUC_PREREQ (4, 3)
    # # define _GL_ATTR_cold _GL_GNUC_PREREQ (4, 3)
    # # define _GL_ATTR_const _GL_GNUC_PREREQ (2, 95)
    # # define _GL_ATTR_deprecated _GL_GNUC_PREREQ (3, 1)
    # # define _GL_ATTR_diagnose_if 0
    # # define _GL_ATTR_error _GL_GNUC_PREREQ (4, 3)
    # # define _GL_ATTR_externally_visible _GL_GNUC_PREREQ (4, 1)
    # # define _GL_ATTR_fallthrough _GL_GNUC_PREREQ (7, 0)
    # # define _GL_ATTR_format _GL_GNUC_PREREQ (2, 7)
    # # define _GL_ATTR_leaf _GL_GNUC_PREREQ (4, 6)
    # # define _GL_ATTR_malloc _GL_GNUC_PREREQ (3, 0)
    # # ifdef _ICC
    # #  define _GL_ATTR_may_alias 0
    # # else
    # #  define _GL_ATTR_may_alias _GL_GNUC_PREREQ (3, 3)
    # # endif
    # # define _GL_ATTR_noinline _GL_GNUC_PREREQ (3, 1)
    # # define _GL_ATTR_nonnull _GL_GNUC_PREREQ (3, 3)
    # # define _GL_ATTR_nonstring _GL_GNUC_PREREQ (8, 0)
    # # define _GL_ATTR_nothrow _GL_GNUC_PREREQ (3, 3)
    # # define _GL_ATTR_packed _GL_GNUC_PREREQ (2, 7)
    # # define _GL_ATTR_pure _GL_GNUC_PREREQ (2, 96)
    # # define _GL_ATTR_returns_nonnull _GL_GNUC_PREREQ (4, 9)
    # # define _GL_ATTR_sentinel _GL_GNUC_PREREQ (4, 0)
    # # define _GL_ATTR_unused _GL_GNUC_PREREQ (2, 7)
    # # define _GL_ATTR_warn_unused_result _GL_GNUC_PREREQ (3, 4)
    # #endif
    #
    # #ifdef __has_c_attribute
    # # define _GL_HAS_C_ATTRIBUTE(attr) __has_c_attribute (__##attr##__)
    # #else
    # # define _GL_HAS_C_ATTRIBUTE(attr) 0
    # #endif
    #
    #
    # /* _GL_ATTRIBUTE_ALLOC_SIZE ((N)) declares that the Nth argument of the function
    #    is the size of the returned memory block.
    #    _GL_ATTRIBUTE_ALLOC_SIZE ((M, N)) declares that the Mth argument multiplied
    #    by the Nth argument of the function is the size of the returned memory block.
    #  */
    # /* Applies to: function, pointer to function, function types.  */
    # #if _GL_HAS_ATTRIBUTE (alloc_size)
    # # define _GL_ATTRIBUTE_ALLOC_SIZE(args) __attribute__ ((__alloc_size__ args))
    # #else
    # # define _GL_ATTRIBUTE_ALLOC_SIZE(args)
    # #endif
    #
    # /* _GL_ATTRIBUTE_ALWAYS_INLINE tells that the compiler should always inline the
    #    function and report an error if it cannot do so.  */
    # /* Applies to: function.  */
    # #if _GL_HAS_ATTRIBUTE (always_inline)
    # # define _GL_ATTRIBUTE_ALWAYS_INLINE __attribute__ ((__always_inline__))
    # #else
    # # define _GL_ATTRIBUTE_ALWAYS_INLINE
    # #endif
    #
    # /* _GL_ATTRIBUTE_ARTIFICIAL declares that the function is not important to show
    #     in stack traces when debugging.  The compiler should omit the function from
    #     stack traces.  */
    # /* Applies to: function.  */
    # #if _GL_HAS_ATTRIBUTE (artificial)
    # # define _GL_ATTRIBUTE_ARTIFICIAL __attribute__ ((__artificial__))
    # #else
    # # define _GL_ATTRIBUTE_ARTIFICIAL
    # #endif
    #
    # /* _GL_ATTRIBUTE_COLD declares that the function is rarely executed.  */
    # /* Applies to: functions.  */
    # /* Avoid __attribute__ ((cold)) on MinGW; see thread starting at
    #    <https://lists.gnu.org/r/emacs-devel/2019-04/msg01152.html>.
    #    Also, Oracle Studio 12.6 requires 'cold' not '__cold__'.  */
    # #if _GL_HAS_ATTRIBUTE (cold) && !defined __MINGW32__
    # # ifndef __SUNPRO_C
    # #  define _GL_ATTRIBUTE_COLD __attribute__ ((__cold__))
    # # else
    # #  define _GL_ATTRIBUTE_COLD __attribute__ ((cold))
    # # endif
    # #else
    # # define _GL_ATTRIBUTE_COLD
    # #endif
    #
    # /* _GL_ATTRIBUTE_CONST declares that it is OK for a compiler to omit duplicate
    #    calls to the function with the same arguments.
    #    This attribute is safe for a function that neither depends on nor affects
    #    observable state, and always returns exactly once - e.g., does not loop
    #    forever, and does not call longjmp.
    #    (This attribute is stricter than _GL_ATTRIBUTE_PURE.)  */
    # /* Applies to: functions.  */
    # #if _GL_HAS_ATTRIBUTE (const)
    # # define _GL_ATTRIBUTE_CONST __attribute__ ((__const__))
    # #else
    # # define _GL_ATTRIBUTE_CONST
    # #endif
    #
    # /* _GL_ATTRIBUTE_DEALLOC (F, I) declares that the function returns pointers
    #    that can be freed by passing them as the Ith argument to the
    #    function F.
    #    _GL_ATTRIBUTE_DEALLOC_FREE declares that the function returns pointers that
    #    can be freed via 'free'; it can be used only after declaring 'free'.  */
    # /* Applies to: functions.  Cannot be used on inline functions.  */
    # #if _GL_GNUC_PREREQ (11, 0)
    # # define _GL_ATTRIBUTE_DEALLOC(f, i) __attribute__ ((__malloc__ (f, i)))
    # #else
    # # define _GL_ATTRIBUTE_DEALLOC(f, i)
    # #endif
    # #define _GL_ATTRIBUTE_DEALLOC_FREE _GL_ATTRIBUTE_DEALLOC (free, 1)
    _GL_ATTRIBUTE_DEALLOC_FREE = "_GL_ATTRIBUTE_DEALLOC (free, 1)"
    #
    # /* _GL_ATTRIBUTE_DEPRECATED: Declares that an entity is deprecated.
    #    The compiler may warn if the entity is used.  */
    # /* Applies to:
    #      - function, variable,
    #      - struct, union, struct/union member,
    #      - enumeration, enumeration item,
    #      - typedef,
    #    in C++ also: namespace, class, template specialization.  */
    # #if _GL_HAS_C_ATTRIBUTE (deprecated)
    # # define _GL_ATTRIBUTE_DEPRECATED [[__deprecated__]]
    # #elif _GL_HAS_ATTRIBUTE (deprecated)
    # # define _GL_ATTRIBUTE_DEPRECATED __attribute__ ((__deprecated__))
    # #else
    # # define _GL_ATTRIBUTE_DEPRECATED
    # #endif
    #
    # /* _GL_ATTRIBUTE_ERROR(msg) requests an error if a function is called and
    #    the function call is not optimized away.
    #    _GL_ATTRIBUTE_WARNING(msg) requests a warning if a function is called and
    #    the function call is not optimized away.  */
    # /* Applies to: functions.  */
    # #if _GL_HAS_ATTRIBUTE (error)
    # # define _GL_ATTRIBUTE_ERROR(msg) __attribute__ ((__error__ (msg)))
    # # define _GL_ATTRIBUTE_WARNING(msg) __attribute__ ((__warning__ (msg)))
    # #elif _GL_HAS_ATTRIBUTE (diagnose_if)
    # # define _GL_ATTRIBUTE_ERROR(msg) __attribute__ ((__diagnose_if__ (1, msg, "error")))
    # # define _GL_ATTRIBUTE_WARNING(msg) __attribute__ ((__diagnose_if__ (1, msg, "warning")))
    # #else
    # # define _GL_ATTRIBUTE_ERROR(msg)
    # # define _GL_ATTRIBUTE_WARNING(msg)
    # #endif
    #
    # /* _GL_ATTRIBUTE_EXTERNALLY_VISIBLE declares that the entity should remain
    #    visible to debuggers etc., even with '-fwhole-program'.  */
    # /* Applies to: functions, variables.  */
    # #if _GL_HAS_ATTRIBUTE (externally_visible)
    # # define _GL_ATTRIBUTE_EXTERNALLY_VISIBLE __attribute__ ((externally_visible))
    # #else
    # # define _GL_ATTRIBUTE_EXTERNALLY_VISIBLE
    # #endif
    #
    # /* _GL_ATTRIBUTE_FALLTHROUGH declares that it is not a programming mistake if
    #    the control flow falls through to the immediately following 'case' or
    #    'default' label.  The compiler should not warn in this case.  */
    # /* Applies to: Empty statement (;), inside a 'switch' statement.  */
    # /* Always expands to something.  */
    # #if _GL_HAS_C_ATTRIBUTE (fallthrough)
    # # define _GL_ATTRIBUTE_FALLTHROUGH [[__fallthrough__]]
    # #elif _GL_HAS_ATTRIBUTE (fallthrough)
    # # define _GL_ATTRIBUTE_FALLTHROUGH __attribute__ ((__fallthrough__))
    # #else
    # # define _GL_ATTRIBUTE_FALLTHROUGH ((void) 0)
    # #endif
    #
    # /* _GL_ATTRIBUTE_FORMAT ((ARCHETYPE, STRING-INDEX, FIRST-TO-CHECK))
    #    declares that the STRING-INDEXth function argument is a format string of
    #    style ARCHETYPE, which is one of:
    #      printf, gnu_printf
    #      scanf, gnu_scanf,
    #      strftime, gnu_strftime,
    #      strfmon,
    #    or the same thing prefixed and suffixed with '__'.
    #    If FIRST-TO-CHECK is not 0, arguments starting at FIRST-TO_CHECK
    #    are suitable for the format string.  */
    # /* Applies to: functions.  */
    # #if _GL_HAS_ATTRIBUTE (format)
    # # define _GL_ATTRIBUTE_FORMAT(spec) __attribute__ ((__format__ spec))
    # #else
    # # define _GL_ATTRIBUTE_FORMAT(spec)
    # #endif
    #
    # /* _GL_ATTRIBUTE_LEAF declares that if the function is called from some other
    #    compilation unit, it executes code from that unit only by return or by
    #    exception handling.  This declaration lets the compiler optimize that unit
    #    more aggressively.  */
    # /* Applies to: functions.  */
    # #if _GL_HAS_ATTRIBUTE (leaf)
    # # define _GL_ATTRIBUTE_LEAF __attribute__ ((__leaf__))
    # #else
    # # define _GL_ATTRIBUTE_LEAF
    # #endif
    #
    # /* _GL_ATTRIBUTE_MALLOC declares that the function returns a pointer to freshly
    #    allocated memory.  */
    # /* Applies to: functions.  */
    # #if _GL_HAS_ATTRIBUTE (malloc)
    # # define _GL_ATTRIBUTE_MALLOC __attribute__ ((__malloc__))
    # #else
    # # define _GL_ATTRIBUTE_MALLOC
    # #endif
    #
    # /* _GL_ATTRIBUTE_MAY_ALIAS declares that pointers to the type may point to the
    #    same storage as pointers to other types.  Thus this declaration disables
    #    strict aliasing optimization.  */
    # /* Applies to: types.  */
    # /* Oracle Studio 12.6 mishandles may_alias despite __has_attribute OK.  */
    # #if _GL_HAS_ATTRIBUTE (may_alias) && !defined __SUNPRO_C
    # # define _GL_ATTRIBUTE_MAY_ALIAS __attribute__ ((__may_alias__))
    # #else
    # # define _GL_ATTRIBUTE_MAY_ALIAS
    # #endif
    #
    # /* _GL_ATTRIBUTE_MAYBE_UNUSED declares that it is not a programming mistake if
    #    the entity is not used.  The compiler should not warn if the entity is not
    #    used.  */
    # /* Applies to:
    #      - function, variable,
    #      - struct, union, struct/union member,
    #      - enumeration, enumeration item,
    #      - typedef,
    #    in C++ also: class.  */
    # /* In C++ and C2x, this is spelled [[__maybe_unused__]].
    #    GCC's syntax is __attribute__ ((__unused__)).
    #    clang supports both syntaxes.  */
    # #if _GL_HAS_C_ATTRIBUTE (maybe_unused)
    # # define _GL_ATTRIBUTE_MAYBE_UNUSED [[__maybe_unused__]]
    # #else
    # # define _GL_ATTRIBUTE_MAYBE_UNUSED _GL_ATTRIBUTE_UNUSED
    # #endif
    # /* Alternative spelling of this macro, for convenience.  */
    # #define _GL_UNUSED _GL_ATTRIBUTE_MAYBE_UNUSED
    _GL_UNUSED = "_GL_ATTRIBUTE_MAYBE_UNUSED"
    # /* Earlier spellings of this macro.  */
    # #define _UNUSED_PARAMETER_ _GL_ATTRIBUTE_MAYBE_UNUSED
    _UNUSED_PARAMETER_ = "_GL_ATTRIBUTE_MAYBE_UNUSED"
    #
    # /* _GL_ATTRIBUTE_NODISCARD declares that the caller of the function should not
    #    discard the return value.  The compiler may warn if the caller does not use
    #    the return value, unless the caller uses something like ignore_value.  */
    # /* Applies to: function, enumeration, class.  */
    # #if _GL_HAS_C_ATTRIBUTE (nodiscard)
    # # define _GL_ATTRIBUTE_NODISCARD [[__nodiscard__]]
    # #elif _GL_HAS_ATTRIBUTE (warn_unused_result)
    # # define _GL_ATTRIBUTE_NODISCARD __attribute__ ((__warn_unused_result__))
    # #else
    # # define _GL_ATTRIBUTE_NODISCARD
    # #endif
    #
    # /* _GL_ATTRIBUTE_NOINLINE tells that the compiler should not inline the
    #    function.  */
    # /* Applies to: functions.  */
    # #if _GL_HAS_ATTRIBUTE (noinline)
    # # define _GL_ATTRIBUTE_NOINLINE __attribute__ ((__noinline__))
    # #else
    # # define _GL_ATTRIBUTE_NOINLINE
    # #endif
    #
    # /* _GL_ATTRIBUTE_NONNULL ((N1, N2,...)) declares that the arguments N1, N2,...
    #    must not be NULL.
    #    _GL_ATTRIBUTE_NONNULL () declares that all pointer arguments must not be
    #    null.  */
    # /* Applies to: functions.  */
    # #if _GL_HAS_ATTRIBUTE (nonnull)
    # # define _GL_ATTRIBUTE_NONNULL(args) __attribute__ ((__nonnull__ args))
    # #else
    # # define _GL_ATTRIBUTE_NONNULL(args)
    # #endif
    #
    # /* _GL_ATTRIBUTE_NONSTRING declares that the contents of a character array is
    #    not meant to be NUL-terminated.  */
    # /* Applies to: struct/union members and variables that are arrays of element
    #    type '[[un]signed] char'.  */
    # #if _GL_HAS_ATTRIBUTE (nonstring)
    # # define _GL_ATTRIBUTE_NONSTRING __attribute__ ((__nonstring__))
    # #else
    # # define _GL_ATTRIBUTE_NONSTRING
    # #endif
    #
    # /* There is no _GL_ATTRIBUTE_NORETURN; use _Noreturn instead.  */
    #
    # /* _GL_ATTRIBUTE_NOTHROW declares that the function does not throw exceptions.
    #  */
    # /* Applies to: functions.  */
    # #if _GL_HAS_ATTRIBUTE (nothrow) && !defined __cplusplus
    # # define _GL_ATTRIBUTE_NOTHROW __attribute__ ((__nothrow__))
    # #else
    # # define _GL_ATTRIBUTE_NOTHROW
    # #endif
    #
    # /* _GL_ATTRIBUTE_PACKED declares:
    #    For struct members: The member has the smallest possible alignment.
    #    For struct, union, class: All members have the smallest possible alignment,
    #    minimizing the memory required.  */
    # /* Applies to: struct members, struct, union,
    #    in C++ also: class.  */
    # #if _GL_HAS_ATTRIBUTE (packed)
    # # define _GL_ATTRIBUTE_PACKED __attribute__ ((__packed__))
    # #else
    # # define _GL_ATTRIBUTE_PACKED
    # #endif
    #
    # /* _GL_ATTRIBUTE_PURE declares that It is OK for a compiler to omit duplicate
    #    calls to the function with the same arguments if observable state is not
    #    changed between calls.
    #    This attribute is safe for a function that does not affect
    #    observable state, and always returns exactly once.
    #    (This attribute is looser than _GL_ATTRIBUTE_CONST.)  */
    # /* Applies to: functions.  */
    # #if _GL_HAS_ATTRIBUTE (pure)
    # # define _GL_ATTRIBUTE_PURE __attribute__ ((__pure__))
    # #else
    # # define _GL_ATTRIBUTE_PURE
    # #endif
    #
    # /* _GL_ATTRIBUTE_RETURNS_NONNULL declares that the function's return value is
    #    a non-NULL pointer.  */
    # /* Applies to: functions.  */
    # #if _GL_HAS_ATTRIBUTE (returns_nonnull)
    # # define _GL_ATTRIBUTE_RETURNS_NONNULL __attribute__ ((__returns_nonnull__))
    # #else
    # # define _GL_ATTRIBUTE_RETURNS_NONNULL
    # #endif
    #
    # /* _GL_ATTRIBUTE_SENTINEL(pos) declares that the variadic function expects a
    #    trailing NULL argument.
    #    _GL_ATTRIBUTE_SENTINEL () - The last argument is NULL (requires C99).
    #    _GL_ATTRIBUTE_SENTINEL ((N)) - The (N+1)st argument from the end is NULL.  */
    # /* Applies to: functions.  */
    # #if _GL_HAS_ATTRIBUTE (sentinel)
    # # define _GL_ATTRIBUTE_SENTINEL(pos) __attribute__ ((__sentinel__ pos))
    # #else
    # # define _GL_ATTRIBUTE_SENTINEL(pos)
    # #endif
    #
    # /* A helper macro.  Don't use it directly.  */
    # #if _GL_HAS_ATTRIBUTE (unused)
    # # define _GL_ATTRIBUTE_UNUSED __attribute__ ((__unused__))
    # #else
    # # define _GL_ATTRIBUTE_UNUSED
    # #endif
    #
    #
    # /* _GL_UNUSED_LABEL; declares that it is not a programming mistake if the
    #    immediately preceding label is not used.  The compiler should not warn
    #    if the label is not used.  */
    # /* Applies to: label (both in C and C++).  */
    # /* Note that g++ < 4.5 does not support the '__attribute__ ((__unused__)) ;'
    #    syntax.  But clang does.  */
    # #if !(defined __cplusplus && !_GL_GNUC_PREREQ (4, 5)) || defined __clang__
    # # define _GL_UNUSED_LABEL _GL_ATTRIBUTE_UNUSED
    # #else
    # # define _GL_UNUSED_LABEL
    # #endif
    #
    #
    # /* Please see the Gnulib manual for how to use these macros.
    #
    #    Suppress extern inline with HP-UX cc, as it appears to be broken; see
    #    <https://lists.gnu.org/r/bug-texinfo/2013-02/msg00030.html>.
    #
    #    Suppress extern inline with Sun C in standards-conformance mode, as it
    #    mishandles inline functions that call each other.  E.g., for 'inline void f
    #    (void) { } inline void g (void) { f (); }', c99 incorrectly complains
    #    'reference to static identifier "f" in extern inline function'.
    #    This bug was observed with Oracle Developer Studio 12.6
    #    (Sun C 5.15 SunOS_sparc 2017/05/30).
    #
    #    Suppress extern inline (with or without __attribute__ ((__gnu_inline__)))
    #    on configurations that mistakenly use 'static inline' to implement
    #    functions or macros in standard C headers like <ctype.h>.  For example,
    #    if isdigit is mistakenly implemented via a static inline function,
    #    a program containing an extern inline function that calls isdigit
    #    may not work since the C standard prohibits extern inline functions
    #    from calling static functions (ISO C 99 section 6.7.4.(3).
    #    This bug is known to occur on:
    #
    #      OS X 10.8 and earlier; see:
    #      https://lists.gnu.org/r/bug-gnulib/2012-12/msg00023.html
    #
    #      DragonFly; see
    #      http://muscles.dragonflybsd.org/bulk/clang-master-potential/20141111_102002/logs/ah-tty-0.3.12.log
    #
    #      FreeBSD; see:
    #      https://lists.gnu.org/r/bug-gnulib/2014-07/msg00104.html
    #
    #    OS X 10.9 has a macro __header_inline indicating the bug is fixed for C and
    #    for clang but remains for g++; see <https://trac.macports.org/ticket/41033>.
    #    Assume DragonFly and FreeBSD will be similar.
    #
    #    GCC 4.3 and above with -std=c99 or -std=gnu99 implements ISO C99
    #    inline semantics, unless -fgnu89-inline is used.  It defines a macro
    #    __GNUC_STDC_INLINE__ to indicate this situation or a macro
    #    __GNUC_GNU_INLINE__ to indicate the opposite situation.
    #    GCC 4.2 with -std=c99 or -std=gnu99 implements the GNU C inline
    #    semantics but warns, unless -fgnu89-inline is used:
    #      warning: C99 inline functions are not supported; using GNU89
    #      warning: to disable this warning use -fgnu89-inline or the gnu_inline function attribute
    #    It defines a macro __GNUC_GNU_INLINE__ to indicate this situation.
    #  */
    # #if (((defined __APPLE__ && defined __MACH__) \
    #       || defined __DragonFly__ || defined __FreeBSD__) \
    #      && (defined __header_inline \
    #          ? (defined __cplusplus && defined __GNUC_STDC_INLINE__ \
    #             && ! defined __clang__) \
    #          : ((! defined _DONT_USE_CTYPE_INLINE_ \
    #              && (defined __GNUC__ || defined __cplusplus)) \
    #             || (defined _FORTIFY_SOURCE && 0 < _FORTIFY_SOURCE \
    #                 && defined __GNUC__ && ! defined __cplusplus))))
    # # define _GL_EXTERN_INLINE_STDHEADER_BUG
    # #endif
    # #if ((__GNUC__ \
    #       ? defined __GNUC_STDC_INLINE__ && __GNUC_STDC_INLINE__ \
    #       : (199901L <= __STDC_VERSION__ \
    #          && !defined __HP_cc \
    #          && !defined __PGI \
    #          && !(defined __SUNPRO_C && __STDC__))) \
    #      && !defined _GL_EXTERN_INLINE_STDHEADER_BUG)
    # # define _GL_INLINE inline
    # # define _GL_EXTERN_INLINE extern inline
    # # define _GL_EXTERN_INLINE_IN_USE
    # #elif (2 < __GNUC__ + (7 <= __GNUC_MINOR__) && !defined __STRICT_ANSI__ \
    #        && !defined _GL_EXTERN_INLINE_STDHEADER_BUG)
    # # if defined __GNUC_GNU_INLINE__ && __GNUC_GNU_INLINE__
    #    /* __gnu_inline__ suppresses a GCC 4.2 diagnostic.  */
    # #  define _GL_INLINE extern inline __attribute__ ((__gnu_inline__))
    # # else
    # #  define _GL_INLINE extern inline
    # # endif
    # # define _GL_EXTERN_INLINE extern
    # # define _GL_EXTERN_INLINE_IN_USE
    # #else
    # # define _GL_INLINE _GL_UNUSED static
    # # define _GL_EXTERN_INLINE _GL_UNUSED static
    # #endif
    #
    # /* In GCC 4.6 (inclusive) to 5.1 (exclusive),
    #    suppress bogus "no previous prototype for 'FOO'"
    #    and "no previous declaration for 'FOO'" diagnostics,
    #    when FOO is an inline function in the header; see
    #    <https://gcc.gnu.org/bugzilla/show_bug.cgi?id=54113> and
    #    <https://gcc.gnu.org/bugzilla/show_bug.cgi?id=63877>.  */
    # #if __GNUC__ == 4 && 6 <= __GNUC_MINOR__
    # # if defined __GNUC_STDC_INLINE__ && __GNUC_STDC_INLINE__
    # #  define _GL_INLINE_HEADER_CONST_PRAGMA
    # # else
    # #  define _GL_INLINE_HEADER_CONST_PRAGMA \
    #      _Pragma ("GCC diagnostic ignored \"-Wsuggest-attribute=const\"")
    # # endif
    # # define _GL_INLINE_HEADER_BEGIN \
    #     _Pragma ("GCC diagnostic push") \
    #     _Pragma ("GCC diagnostic ignored \"-Wmissing-prototypes\"") \
    #     _Pragma ("GCC diagnostic ignored \"-Wmissing-declarations\"") \
    #     _GL_INLINE_HEADER_CONST_PRAGMA
    # # define _GL_INLINE_HEADER_END \
    #     _Pragma ("GCC diagnostic pop")
    # #else
    # # define _GL_INLINE_HEADER_BEGIN
    # # define _GL_INLINE_HEADER_END
    # #endif
    #
    # /* Define to `__inline__' or `__inline' if that's what the C compiler
    #    calls it, or to nothing if 'inline' is not supported under any name.  */
    # #ifndef __cplusplus
    # /* #undef inline */
    # #endif
    #
    # /* Work around a bug in Apple GCC 4.0.1 build 5465: In C99 mode, it supports
    #    the ISO C 99 semantics of 'extern inline' (unlike the GNU C semantics of
    #    earlier versions), but does not display it by setting __GNUC_STDC_INLINE__.
    #    __APPLE__ && __MACH__ test for Mac OS X.
    #    __APPLE_CC__ tests for the Apple compiler and its version.
    #    __STDC_VERSION__ tests for the C99 mode.  */
    # #if defined __APPLE__ && defined __MACH__ && __APPLE_CC__ >= 5465 && !defined __cplusplus && __STDC_VERSION__ >= 199901L && !defined __GNUC_STDC_INLINE__
    # # define __GNUC_STDC_INLINE__ 1
    # #endif
    #
    # /* Define to a type if <wchar.h> does not define. */
    # /* #undef mbstate_t */
    #
    # /* _GL_CMP (n1, n2) performs a three-valued comparison on n1 vs. n2, where
    #    n1 and n2 are expressions without side effects, that evaluate to real
    #    numbers (excluding NaN).
    #    It returns
    #      1  if n1 > n2
    #      0  if n1 == n2
    #      -1 if n1 < n2
    #    The naïve code   (n1 > n2 ? 1 : n1 < n2 ? -1 : 0)  produces a conditional
    #    jump with nearly all GCC versions up to GCC 10.
    #    This variant     (n1 < n2 ? -1 : n1 > n2)  produces a conditional with many
    #    GCC versions up to GCC 9.
    #    The better code  (n1 > n2) - (n1 < n2)  from Hacker's Delight § 2-9
    #    avoids conditional jumps in all GCC versions >= 3.4.  */
    # #define _GL_CMP(n1, n2) (((n1) > (n2)) - ((n1) < (n2)))
    @staticmethod
    def _GL_CMP(n1, n2):
        return int(n1 > n2) - int(n1 < n2)
    #
    #
    # /* Define to the real name of the mktime_internal function. */
    # /* #undef mktime_internal */
    #
    # /* Define to `int' if <sys/types.h> does not define. */
    # /* #undef mode_t */
    #
    # /* Define to the name of the strftime replacement function. */
    # #define my_strftime nstrftime
    my_strftime = "nstrftime"
    #
    # /* Define to the type of st_nlink in struct stat, or a supertype. */
    # /* #undef nlink_t */
    #
    # /* Define as a signed integer type capable of holding a process identifier. */
    # /* #undef pid_t */
    #
    # /* Define to rpl_re_comp if the replacement should be used. */
    # #define re_comp rpl_re_comp
    re_comp = "rpl_re_comp"
    #
    # /* Define to rpl_re_compile_fastmap if the replacement should be used. */
    # #define re_compile_fastmap rpl_re_compile_fastmap
    re_compile_fastmap = "rpl_re_compile_fastmap"
    #
    # /* Define to rpl_re_compile_pattern if the replacement should be used. */
    # #define re_compile_pattern rpl_re_compile_pattern
    re_compile_pattern = "rpl_re_compile_pattern"
    #
    # /* Define to rpl_re_exec if the replacement should be used. */
    # #define re_exec rpl_re_exec
    re_exec = "rpl_re_exec"
    #
    # /* Define to rpl_re_match if the replacement should be used. */
    # #define re_match rpl_re_match
    re_match = "rpl_re_match"
    #
    # /* Define to rpl_re_match_2 if the replacement should be used. */
    # #define re_match_2 rpl_re_match_2
    re_match_2 = "rpl_re_match_2"
    #
    # /* Define to rpl_re_search if the replacement should be used. */
    # #define re_search rpl_re_search
    re_search = "rpl_re_search"
    #
    # /* Define to rpl_re_search_2 if the replacement should be used. */
    # #define re_search_2 rpl_re_search_2
    re_search_2 = "rpl_re_search_2"
    #
    # /* Define to rpl_re_set_registers if the replacement should be used. */
    # #define re_set_registers rpl_re_set_registers
    re_set_registers = "rpl_re_set_registers"
    #
    # /* Define to rpl_re_set_syntax if the replacement should be used. */
    # #define re_set_syntax rpl_re_set_syntax
    re_set_syntax = "rpl_re_set_syntax"
    #
    # /* Define to rpl_re_syntax_options if the replacement should be used. */
    # #define re_syntax_options rpl_re_syntax_options
    re_syntax_options = "rpl_re_syntax_options"
    #
    # /* Define to rpl_regcomp if the replacement should be used. */
    # #define regcomp rpl_regcomp
    regcomp = "rpl_regcomp"
    #
    # /* Define to rpl_regerror if the replacement should be used. */
    # #define regerror rpl_regerror
    regerror = "rpl_regerror"
    #
    # /* Define to rpl_regexec if the replacement should be used. */
    # #define regexec rpl_regexec
    regexec = "rpl_regexec"
    #
    # /* Define to rpl_regfree if the replacement should be used. */
    # #define regfree rpl_regfree
    regfree = "rpl_regfree"
    #
    # /* Define to the equivalent of the C99 'restrict' keyword, or to
    #    nothing if this is not supported.  Do not define if restrict is
    #    supported only directly.  */
    # #define restrict __restrict__
    restrict = "__restrict__"
    # /* Work around a bug in older versions of Sun C++, which did not
    #    #define __restrict__ or support _Restrict or __restrict__
    #    even though the corresponding Sun C compiler ended up with
    #    "#define restrict _Restrict" or "#define restrict __restrict__"
    #    in the previous line.  This workaround can be removed once
    #    we assume Oracle Developer Studio 12.5 (2016) or later.  */
    # #if defined __SUNPRO_CC && !defined __RESTRICT && !defined __restrict__
    # # define _Restrict
    # # define __restrict__
    # #endif
    #
    # /* type to use in place of socklen_t if not defined */
    # /* #undef socklen_t */
    #
    # /* Define as a signed type of the same size as size_t. */
    # /* #undef ssize_t */
    #
    # /* Define to enable asynchronous subprocesses. */
    # #define subprocesses 1
    subprocesses = 1
    #
    # /* Define to __typeof__ if your compiler spells it that way. */
    # /* #undef typeof */
    #
    # /* Define as `fork' if `vfork' does not work. */
    # /* #undef vfork */
    #
    # #include <conf_post.h>
    #
    # #endif /* EMACS_CONFIG_H */
    #
    # /*
    # Local Variables:
    # mode: c
    # End:
    # */
