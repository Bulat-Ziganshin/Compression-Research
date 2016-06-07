
#define mymin(a,b)               ((a)<(b)? (a) : (b))
#define mymax(a,b)               ((a)>(b)? (a) : (b))

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
    char* endptr;
    auto res = strtoll (argv + strlen(prefix), &endptr, 10);
    if (*endptr)  return false;
    *option  =  T(res);
    return true;
}

// Parse integer list option starting with given prefix, return true if option was recognized
// Option string shoould be "[+/-]n,m-k..." which means enable/disable/enable_only channels n and m..k
static inline bool ParseIntList (char *argv, char *prefix, bool *enabled, int size)
{
    if (memcmp (argv, prefix, strlen(prefix)))  return false;
    argv += strlen(prefix);

    // If option was successfully recognized, skip initial +/- and set appropriate editing mode
    bool enable_or_disable = true;
    if (*argv=='+')       argv++;                                           // Enable additional channels
    else if (*argv=='-')  argv++,  enable_or_disable = false;               // Disable some channels
    else                  {for (int i=0; i<size; i++)  enabled[i]=false;}   // Enable only channels listed in this option
    
    // Parse multiple numbers "n" or ranges "n-m", separated by commas;  enable/disable all channels specified by parsed numbers
    for(;;)
    {
        char* endptr;
        auto first = strtoll (argv, &endptr, 10);
        auto second = first;
        if (endptr==argv)  return false;  // no digits
        if (*endptr=='-')  {              // range "n-m"
            second = strtoll (endptr+1, &endptr, 10);
        }
        if (first > second  ||  second >= size)  return false;   // invalid range of numbers
        for (int i=first; i<=second; i++)  
            enabled[i] = enable_or_disable;
        if (*endptr==0)    break; 
        if (*endptr!=',')  return false;
        argv = endptr+1;
    }
    
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
