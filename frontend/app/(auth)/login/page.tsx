"use client";

import { z } from "zod";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import { Eye, EyeOff } from "lucide-react";
import { useState } from "react";
import axios from "axios";
import { useRouter } from "next/navigation";
import toast from "react-hot-toast";
import Link from "next/link";

import { isStrongPassword } from "@/lib/validations";

import { Form, FormControl, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";

const LoginSchema = z.object({
  email: z
    .string()
    .trim()
    .min(1, { message: "Please enter your email address." })
    .email({ message: "Invalid email address." })
    .max(254, { message: "Too long email address." }),
  password: z
    .string()
    .trim()
    .min(8, { message: "Password must be at least 8 characters." })
    .refine(isStrongPassword, { message: "Password must contain a-z, A-Z, 0-9, and !@#$%^&*." }),
});

export default function Login() {
  const [showPassword, setShowPassword] = useState(false);
  const router = useRouter();

  const form = useForm<z.infer<typeof LoginSchema>>({
    resolver: zodResolver(LoginSchema),
    defaultValues: {
      email: "",
      password: "",
    }
  })

  const onSubmit = async (data: z.infer<typeof LoginSchema>) => {
    try {
      // eslint-disable-next-line
      const res = await axios.post("http://localhost:8080/api/login", data, { withCredentials: true });
      toast.success("You're successfully logged in!");
      router.push("/");
    } catch {
      toast.error("Something went wrong");
    }
  }

  return (
    <div className="h-full flex justify-center items-center">
      <div className="flex flex-col w-4/5 md:w-2/5 xl:w-1/4 space-y-4">
        <h1 className="text-2xl font-bold">Welcome Back!</h1>
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
            <FormField
              control={form.control}
              name="email"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Email</FormLabel>
                  <FormControl>
                    <Input {...field}/>
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="password"
              render={({ field }) => (
                <FormItem className="w-full">
                  <FormLabel>Password</FormLabel>
                    <FormControl>
                      <div className="relative w-full">
                        <Input
                          minLength={8}
                          autoComplete="current-password"
                          type={showPassword ? "text" : "password"}
                          {...field}
                        />
                        <Button
                          type="button"
                          variant="none"
                          size="sm"
                          className="absolute right-0 top-1/2 transform -translate-y-1/2"
                          onClick={() => setShowPassword(!showPassword)}
                        >
                          {showPassword ?
                            <Eye color="gray" className="h-5 w-5" /> :
                            <EyeOff color="gray" className="h-5 w-5" />
                          }
                        </Button>
                      </div>
                    </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <Button variant="utopia" size="lg" className="w-full">Login</Button>
          </form>
        </Form>
        <div className="flex justify-center space-x-2">
          <p className="text-center">Don&apos;t have an account? <Link href="/sign-up" className="text-blue-500">Sign up</Link></p>
        </div>
      </div>
    </div>
  );
}
