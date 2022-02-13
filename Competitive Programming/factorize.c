#include <stdio.h>

const char *get_mult_string()
{
    static int first_divisor=1;

    if (first_divisor==1) {
        first_divisor=0;
        return "";
    } else {
        return "\n";
    }

}

int main() {
    int n;
    int j;

    scanf("%d", &n);
    j = 2;
    int power_count=0;

    do {
        if (n % j == 0) {
            power_count++;
            n = n / j;
        }

        else {
            if (power_count>0) {
                printf("%s%d %d", get_mult_string(), j, power_count);
                power_count=0;
            }
            j++;
        }
    }

    while (n > 1);
    if (power_count>0) {
        printf("%s%d %d\n", get_mult_string(), j, power_count);
    }
    return 0;

}
