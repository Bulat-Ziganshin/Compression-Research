// Parse boolean option with names for enabling & disabling, return true if option was recognized
bool ParseBool (char *argv, char *enable, char *disable, bool *option)
{
    if (strcmp(argv, enable)==0)  {*option = true;   return true;}
    if (strcmp(argv,disable)==0)  {*option = false;  return true;}
    return false;
}

// Parse string option starting with given prefix, return true if option was recognized
bool ParseStr (char *argv, char *prefix, char **option)
{
    if (memcmp (argv, prefix, strlen(prefix)))  return false;
    *option  =  argv + strlen(prefix);
    return true;
}

// Parse integer option starting with given prefix, return true if option was recognized
template <typename T>
bool ParseInt (char *argv, char *prefix, T *option)
{
    if (memcmp (argv, prefix, strlen(prefix)))  return false;
    *option  =  atoll (argv + strlen(prefix));
    return true;
}

bool UnknownOption (char *argv, int *error)
{
    if (*argv == '-')  {*error = 1; return true;}
    return false;
}
