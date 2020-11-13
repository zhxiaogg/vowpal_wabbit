package vowpalWabbit;

import org.scijava.nativelib.NativeLoader;
import vowpalWabbit.learner.VWLearners;

import java.io.IOException;

public final class VW {
    static {
        try {
            NativeLoader.loadLibrary("vw_jni");
        } catch (IOException e) {
            throw new RuntimeException("cannot load jni lib", e);
        }
    }

    /**
     * Should not be directly instantiated.
     */
    private VW(){}

    /**
     * This main method only exists to test the library implementation.  To test it just run
     * java -cp target/vw-jni-*-SNAPSHOT.jar vowpalWabbit.VW
     * @param args No args needed.
     * @throws Exception possibly during close.
     */
    public static void main(String[] args) throws Exception {
        VWLearners.create("").close();
        VWLearners.create("--quiet").close();
    }

    public static native String version();
}
