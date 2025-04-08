# Debugging Python Segmentation Faults with GDB (Cross-Platform PyTorch/SB3 Issue)

## Problem Summary

When training a PyTorch + Stable Baselines3 (SB3) model on macOS and then evaluating it on Linux and vice verse, a segmentation fault (SIGSEGV) occurured during model loading. This typically happens due to platform-specific incompatibilities in the pickled model or environment, especially if the SB3 model includes environment wrappers or native objects.

```bash
Loading model from models/mario_ppo_latest.zip
Wrapping the env in a VecTransposeImage.
Segmentation fault (core dumped)
```

## Debug with GDB

To debug the segmentation fault, we can use GDB (GNU Debugger) to get a backtrace of the error. This will help us identify where the issue is occurring.

```bash
ubuntu@ip-10-83-199-16:~/mario-brain$ gdb --args python mario_brain/train.py
GNU gdb (Ubuntu 12.1-0ubuntu1~22.04.2) 12.1
Copyright (C) 2022 Free Software Foundation, Inc.
License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.
Type "show copying" and "show warranty" for details.
This GDB was configured as "x86_64-linux-gnu".
Type "show configuration" for configuration details.
For bug reporting instructions, please see:
<https://www.gnu.org/software/gdb/bugs/>.
Find the GDB manual and other documentation resources online at:
    <http://www.gnu.org/software/gdb/documentation/>.

For help, type "help".
Type "apropos word" to search for commands related to "word"...
Reading symbols from python...
(gdb) run
Starting program: /opt/pytorch/bin/python mario_brain/train.py
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
[New Thread 0x7fff24c00640 (LWP 15226)]
[New Thread 0x7fff24200640 (LWP 15227)]
[New Thread 0x7fff23800640 (LWP 15228)]
[New Thread 0x7fff1ba00640 (LWP 15229)]
[Detaching after vfork from child process 15230]
[Detaching after vfork from child process 15231]
/opt/pytorch/lib/python3.12/site-packages/gymnasium/envs/registration.py:519: DeprecationWarning: WARN: The environment SuperMarioBros-v0 is out of date. You should consider upgrading to version `v3`.
  logger.deprecation(
[New Thread 0x7fff09c00640 (LWP 15237)]
Training on cpu
Loading model from models/mario_ppo_latest.zip
Wrapping the env in a VecTransposeImage.

Thread 1 "python" received signal SIGSEGV, Segmentation fault.
_PyEval_EvalFrameDefault (tstate=<optimized out>, frame=0x7ffff7fb0810, throwflag=<optimized out>) at Python/bytecodes.c:2374
2374	Python/bytecodes.c: No such file or directory.
(gdb) bt
#0  _PyEval_EvalFrameDefault (tstate=<optimized out>, frame=0x7ffff7fb0810, throwflag=<optimized out>) at Python/bytecodes.c:2374
#1  0x0000555555757fb7 in _PyObject_FastCallDictTstate (tstate=0x555555c001f8 <_PyRuntime+459704>, callable=0x7fff0ad72e80, args=0x7fffffffd8f0,
    nargsf=<optimized out>, kwargs=<optimized out>) at Objects/call.c:144
#2  0x0000555555793a7f in _PyObject_Call_Prepend (kwargs=0x7fff212c0a80, args=0x7fff0ac1d640, obj=<optimized out>, callable=0x7fff0ad72e80,
    tstate=0x555555c001f8 <_PyRuntime+459704>) at Objects/call.c:508
#3  slot_tp_init (self=<optimized out>, args=0x7fff0ac1d640, kwds=0x7fff212c0a80) at Objects/typeobject.c:9023
#4  0x0000555555755a7d in type_call (type=<optimized out>, args=0x7fff0ac1d640, kwds=0x7fff212c0a80) at Objects/typeobject.c:1677
#5  0x0000555555796c39 in _PyObject_Call (tstate=0x555555c001f8 <_PyRuntime+459704>, callable=0x55555ccd1fe0, args=0x7fff0ac1d640, kwargs=<optimized out>)
    at Objects/call.c:367
#6  0x000055555576594d in PyCFunction_Call (kwargs=0x7fff212c0a80, args=0x7fff0ac1d640, callable=0x55555ccd1fe0) at Objects/call.c:387
#7  _PyEval_EvalFrameDefault (tstate=<optimized out>, frame=0x7ffff7fb0398, throwflag=<optimized out>) at Python/bytecodes.c:3263
#8  0x00005555558249b9 in PyEval_EvalCode (co=<optimized out>, globals=0x7ffff7bf5980, locals=<optimized out>) at Python/ceval.c:578
#9  0x000055555584b74c in run_eval_code_obj (tstate=0x555555c001f8 <_PyRuntime+459704>, co=0x555555cb67c0, globals=0x7ffff7bf5980, locals=0x7ffff7bf5980)
    at Python/pythonrun.c:1722
#10 0x0000555555846b16 in run_mod (mod=<optimized out>, filename=<optimized out>, globals=0x7ffff7bf5980, locals=0x7ffff7bf5980, flags=<optimized out>,
    arena=<optimized out>) at Python/pythonrun.c:1743
#11 0x000055555585f481 in pyrun_file (fp=fp@entry=0x555555c4d400, filename=filename@entry=0x7ffff7b2fe70, start=start@entry=257,
    globals=globals@entry=0x7ffff7bf5980, locals=locals@entry=0x7ffff7bf5980, closeit=closeit@entry=1, flags=0x7fffffffdde0) at Python/pythonrun.c:1643
#12 0x000055555585eea9 in _PyRun_SimpleFileObject (fp=0x555555c4d400, filename=0x7ffff7b2fe70, closeit=1, flags=0x7fffffffdde0) at Python/pythonrun.c:433
#13 0x000055555585eb47 in _PyRun_AnyFileObject (fp=0x555555c4d400, filename=0x7ffff7b2fe70, closeit=1, flags=0x7fffffffdde0) at Python/pythonrun.c:78
#14 0x0000555555857c34 in pymain_run_file_obj (skip_source_first_line=0, filename=0x7ffff7b2fe70, program_name=0x7ffff7bf5ab0) at Modules/main.c:360
#15 pymain_run_file (config=0x555555ba2dd8 <_PyRuntime+77720>) at Modules/main.c:379
#16 pymain_run_python (exitcode=0x7fffffffddb4) at Modules/main.c:633
#17 Py_RunMain () at Modules/main.c:713
#18 0x000055555580a27d in Py_BytesMain (argc=<optimized out>, argv=<optimized out>) at Modules/main.c:767
#19 0x00007ffff7c29d90 in __libc_start_call_main (main=main@entry=0x55555580a1b0 <main>, argc=argc@entry=2, argv=argv@entry=0x7fffffffe038)
    at ../sysdeps/nptl/libc_start_call_main.h:58
#20 0x00007ffff7c29e40 in __libc_start_main_impl (main=0x55555580a1b0 <main>, argc=2, argv=0x7fffffffe038, init=<optimized out>, fini=<optimized out>,
    rtld_fini=<optimized out>, stack_end=0x7fffffffe028) at ../csu/libc-start.c:392
#21 0x000055555580a0e5 in _start ()
```

Show stactrace of the segmentation fault:

```bash
export PYTHONFAULTHANDLER=1
```

## Root Cause

When using a **custom function** for `learning_rate` (or other SB3 config values), the function gets **serialized via `pickle`** and embedded inside the `data` file of the saved `.zip` model. This is not safe across platforms.

Example:

```python
def linear_schedule(initial_value: float, min_value: float):
    def func(progress_remaining: float) -> float:
        return max(min_value, progress_remaining * initial_value)
    return func

model = PPO("CnnPolicy", env, learning_rate=linear_schedule(1e-5, 1e-6), ...)
```

This fuction is saved into data file of the model, and when loading the model on a different platform, it tries to unpickle the function, which fails with a segmentation fault.

```json
    "learning_rate": {
        ":type:": "<class 'function'>",
        ":serialized:": "gAWV6QIAAAAAAACMF2Nsb3VkcGlja2xlLmNsb3VkcGlja2xllIwOX21ha2VfZnVuY3Rpb26Uk5QoaACMDV9idWlsdGluX3R5cGWUk5SMCENvZGVUeXBllIWUUpQoSwFLAEsASwFLBUsTQyKVApcAdAEAAAAAAAAAAIkCfACJAXoFAACrAgAAAAAAAFMAlE6FlIwDbWF4lIWUjBJwcm9ncmVzc19yZW1haW5pbmeUhZSMLS9ob21lL3VidW50dS9tYXJpby1icmFpbi9tYXJpb19icmFpbi90cmFpbi5weZSMBGZ1bmOUjB1saW5lYXJfc2NoZWR1bGUuPGxvY2Fscz4uZnVuY5RLJUMW+IAA3A8SkDnQHjCwPdEeQNMPQdAIQZRDAJSMDWluaXRpYWxfdmFsdWWUjAltaW5fdmFsdWWUhpQpdJRSlH2UKIwLX19wYWNrYWdlX1+UTowIX19uYW1lX1+UjAhfX21haW5fX5SMCF9fZmlsZV9flGgOdU5OaACMEF9tYWtlX2VtcHR5X2NlbGyUk5QpUpRoHilSlIaUdJRSlGgAjBJfZnVuY3Rpb25fc2V0c3RhdGWUk5RoI32UfZQoaBqMBGZ1bmOUjAxfX3F1YWxuYW1lX1+UjB1saW5lYXJfc2NoZWR1bGUuPGxvY2Fscz4uZnVuY5SMD19fYW5ub3RhdGlvbnNfX5R9lCiMEnByb2dyZXNzX3JlbWFpbmluZ5SMCGJ1aWx0aW5zlIwFZmxvYXSUk5SMBnJldHVybpRoMHWMDl9fa3dkZWZhdWx0c19flE6MDF9fZGVmYXVsdHNfX5ROjApfX21vZHVsZV9flGgbjAdfX2RvY19flE6MC19fY2xvc3VyZV9flGgAjApfbWFrZV9jZWxslJOURz7k+LWI42jxhZRSlGg4Rz6wxvegte2NhZRSlIaUjBdfY2xvdWRwaWNrbGVfc3VibW9kdWxlc5RdlIwLX19nbG9iYWxzX1+UfZR1hpSGUjAu"
    },
```

## Workaround

To avoid this we can override the learning rate while loading the model. This will prevent the custom function from being unpickled.

```python
model = PPO.load(path=LATEST_MODEL_PATH, env=env, device=device, learning_rate=linear_schedule(0.00001, 0.000001))
```
