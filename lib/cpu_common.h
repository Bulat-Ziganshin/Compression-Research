
#define mymin(a,b)               ((a)<(b)? (a) : (b))

// Parse boolean option with names for enabling & disabling, return true if option was recognized
static inline bool ParseBool (char *argv, char *enable, char *disable, bool *option)
{
    if (strcmp(argv, enable)==0)  {*option = true;   return true;}
    if (strcmp(argv,disable)==0)  {*option = false;  return true;}
    return false;
}

// Parse string option starting with given prefix, return true if option was recognized
static inline bool ParseStr (char *argv, char *prefix, char **option)
{
    if (memcmp (argv, prefix, strlen(prefix)))  return false;
    *option  =  argv + strlen(prefix);
    return true;
}

// Parse integer option starting with given prefix, return true if option was recognized
template <typename T>
static inline bool ParseInt (char *argv, char *prefix, T *option)
{
    if (memcmp (argv, prefix, strlen(prefix)))  return false;
    *option  =  atoll (argv + strlen(prefix));
    return true;
}

// Return TRUE and set *error if it's an option (not recognized by all preceding checks)
static inline bool UnknownOption (char *argv, int *error)
{
    if (*argv == '-')  {*error = 1; return true;}
    return false;
}

static inline char* show3 (unsigned long long n, char *buf, const char *prepend="")
{
    char *p = buf + 27+strlen(prepend);
    int i = 4;

    *p = '\0';
    do {
        if (!--i) *--p = ',', i = 3;
        *--p = '0' + (n % 10);
    } while (n /= 10);

    memcpy (p-strlen(prepend), prepend, strlen(prepend));
    return p-strlen(prepend);
}

