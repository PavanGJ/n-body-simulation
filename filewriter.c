void writeMeasureToFile(char* dev, char* execType, char* measureType, float measureValue){
    /*
     *  This subroutine appends measure to file for further interpretation.
     */
    FILE* outputStream;
    char output[256] = OUTPUT_DIR;
    char fileName[10] = "measure.csv";
    
    strcat(output, fileName);
    outputStream = fopen(output, "a+");
    
    fprintf(outputStream, "%s, %s, %s, %d, %f\n", dev, execType, measureType, N_SAMPLES, measureValue);
    
    return;
}
