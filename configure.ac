m4_define([DFMTA_VERSION],
          m4_esyscmd([cat DESCRIPTION | grep Version | cut -b 10- | tr -d '\n']))
AC_INIT([dfmta], [DFMTA_VERSION], [jacques-henri.jourdan@cnrs.fr])

AC_CONFIG_MACRO_DIR([tools])

: ${R_HOME=`R RHOME`}
if test -z "${R_HOME}"; then
  echo "could not determine R_HOME"
  exit 1
fi

AC_DEFUN([AX_GET_R_CONF_VAR],[$1=`"${R_HOME}/bin/R" CMD config $1`])
AX_GET_R_CONF_VAR([CC])
AX_GET_R_CONF_VAR([CFLAGS])
AX_GET_R_CONF_VAR([CPPFLAGS])

AC_LANG([C++])

PKG_CXXFLAGS=
PKG_LIBS=

AC_OPENMP
PKG_CXXFLAGS="${PKG_CXXFLAGS} ${OPENMP_CXXFLAGS}"
PKG_LIBS="${PKG_LIBS} ${OPENMP_CXXFLAGS}"

AC_SUBST(PKG_CXXFLAGS)
AC_SUBST(PKG_LIBS)

AC_CONFIG_FILES([src/Makevars])
AC_OUTPUT
